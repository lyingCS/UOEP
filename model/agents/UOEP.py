import time
import copy
import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import uniform
import copy

from tqdm import tqdm

import utils
from model.agents.BaseRLAgent import BaseRLAgent
from model.agents.DDPG import DDPG
from model.agents.BehaviorDDPG import BehaviorDDPG
from model.bandits.algorithm import BetaBernoulliBandit
from model.diversity.loss import DiversityLoss


class UOEP:
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--episode_batch_size', type=int, default=8,
                            help='episode sample batch size')
        parser.add_argument('--embedding_size', type=int, default=20,
                            help='embedding size')
        parser.add_argument('--batch_size', type=int, default=32,
                            help='training batch size')
        parser.add_argument('--actor_lr', type=float, default=1e-4,
                            help='learning rate for actor')
        parser.add_argument('--critic_lr', type=float, default=1e-4,
                            help='decay rate for critic')
        parser.add_argument('--actor_decay', type=float, default=1e-4,
                            help='learning rate for actor')
        parser.add_argument('--critic_decay', type=float, default=1e-4,
                            help='decay rate for critic')
        parser.add_argument('--target_mitigate_coef', type=float, default=0.01,
                            help='mitigation factor')
        parser.add_argument('--gamma', type=float, default=0.95,
                            help='reward discount')
        parser.add_argument('--n_iter', type=int, nargs='+', default=[2000],
                            help='number of training iterations')
        parser.add_argument('--train_every_n_step', type=int, default=1,
                            help='number of training iterations')
        parser.add_argument('--initial_greedy_epsilon', type=float, default=0.6,
                            help='greedy probability for epsilon greedy exploration')
        parser.add_argument('--final_greedy_epsilon', type=float, default=0.05,
                            help='greedy probability for epsilon greedy exploration')
        parser.add_argument('--elbow_greedy', type=float, default=0.5,
                            help='greedy probability for epsilon greedy exploration')
        parser.add_argument('--check_episode', type=int, default=100,
                            help='number of iterations to check output and evaluate')
        parser.add_argument('--with_eval', action='store_true',
                            help='do evaluation during training')
        parser.add_argument('--save_path', type=str, required=True,
                            help='save path for networks')
        parser.add_argument('--test', type=bool, default=False,
                            help='for test')
        parser.add_argument('--div_l', type=int, default=1,
                            help='div l')
        parser.add_argument('--K', type=int, default=8,
                            help='N_QUANTILES_POLICY')
        parser.add_argument('--N', type=int, default=32,
                            help='N_QUANTILES_CRITIC')
        parser.add_argument('--alpha_cvar_list', type=str, default="",
                            help='alpha cvar list')
        parser.add_argument('--below_alpha_cvar_list', type=str, default="",
                            help='below alpha cvar list')
        parser.add_argument('--risk_factor_rou', type=float, default=0.5,
                            help='risk_factor_rou')
        parser.add_argument('--reg_weight', type=float, default=2,
                            help='reg loss weight')
        return parser

    def __init__(self, args, facades):
        self.diversity_loss = DiversityLoss(args.div_l)
        self.episode_batch_size = args.episode_batch_size
        self.device = args.device
        self.batch_size = args.batch_size
        self.embedding_size = args.embedding_size
        self.population_size = len(facades)
        self.save_path = args.save_path
        self.gamma = args.gamma
        self.n_iter = [0] + args.n_iter
        self.train_every_n_step = args.train_every_n_step
        self.check_episode = args.check_episode
        self.with_eval = args.with_eval
        self.diversity_importance = 0
        self.save_path = args.save_path
        self.exploration_scheduler = utils.LinearScheduler(int(sum(args.n_iter) * args.elbow_greedy),
                                                           args.final_greedy_epsilon,
                                                           initial_p=args.initial_greedy_epsilon)

        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.actor_decay = args.actor_decay
        self.critic_decay = args.critic_decay

        self.population = []
        self.population_target = []
        self.facades = facades
        self.buffer_facade = self.facades[0]
        self.diversity_weight = args.diversity_weight
        population_params = []

        for i in range(self.population_size):
            actor = facades[i].actor
            actor_target = copy.deepcopy(actor)
            population_params = population_params + list(actor.parameters())
            self.population.append(actor)
            self.population_target.append(actor_target)

        self.critic = self.buffer_facade.critic
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr,
                                                 weight_decay=args.critic_decay)

        self.population_optimizer = torch.optim.Adam(population_params, lr=args.actor_lr, weight_decay=args.actor_decay)
        self.tau = args.target_mitigate_coef
        self.test_phase = args.test
        self.choice_history = []
        if len(self.n_iter) == 1 and not self.test_phase:
            with open(self.save_path + ".report", 'w') as outfile:
                outfile.write(f"{args}\n")
        self.K = args.K
        self.N = args.N
        self.alpha_cvar_list = list(map(eval, args.alpha_cvar_list.split('_')))
        self.below_alpha_cvar_list = list(map(eval, args.below_alpha_cvar_list.split('_')))

        self.distr_taus_risks = [uniform.Uniform(b_ac, ac)
                                for b_ac, ac in zip(self.below_alpha_cvar_list, self.alpha_cvar_list)]
        self.distr_taus_uniform = uniform.Uniform(0., 1.)
        self.risk_factor_rou = args.risk_factor_rou
        self.distr_taus_risks = [uniform.Uniform(0., 1.) for _ in range(self.population_size)]
        population_params = []
        for i in range(self.population_size):
            actor = facades[i].actor
            population_params = population_params + list(actor.parameters())
        self.reg_weight = args.reg_weight

    def action_before_train(self):
        '''
        Action before training:
        - facade setup:
            - buffer setup
        - run random episodes to build-up the initial buffer
        '''
        for i in range(self.population_size):
            self.facades[i].initialize_train()  # buffer setup
        self.buffer = self.buffer_facade.buffer
        prepare_step = 0
        # random explore before training
        initial_epsilon = 1.0
        observations = [self.facades[i].reset_env({"batch_size": self.episode_batch_size})
                        for i in range(self.population_size)]
        while not self.buffer_facade.is_training_available:
            observations = self.run_episode_step(0, initial_epsilon, observations, True)
            prepare_step += 1
        # training records
        self.training_history = {"critic_loss": [], "actor_loss": [], "diversity_loss": [], "behavior_loss": []}
        print(f"Total {prepare_step} prepare steps")

    def run_episode_step(self, *episode_args):
        '''
        One step of interaction
        '''
        episode_iter, epsilon, observations, do_buffer_update = episode_args
        next_observations = []
        with torch.no_grad():
            # sample action
            # policy_output = self.facade.apply_policy(observation, self.actor, epsilon, do_explore=False)
            for i, observation in enumerate(observations):
                policy_output = self.facades[i].apply_policy(observation, self.population[i], epsilon, do_explore=True)
                # apply action on environment and update replay buffer
                next_observation, reward, done, info = self.facades[i].env_step(policy_output)
                # update replay buffer
                next_observations.append(next_observation)
                if do_buffer_update:
                    self.buffer_facade.update_buffer(observation, policy_output, reward, done, next_observation, info)
        return next_observations

    def action_after_train(self):
        [self.facades[i].stop_env() for i in range(self.population_size)]

    def compute_population_loss(self, observation):
        # embeddings = [self.buffer_facade.apply_policy(observation, agent)['action_emb'].flatten() for agent in self.population]
        embeddings = [agent(observation)['action_emb'].flatten() for agent in self.population]
        embeddings = torch.stack(embeddings)
        return self.diversity_loss(embeddings)

    def set_train(self):
        for actor in self.population:
            actor.train()

    def log_iteration(self, step):
        episode_report, train_report, reward_mean = self.get_report()
        log_str = f"step: {step} @ episode report: {episode_report} @ step loss: {train_report}\n"
        with open(self.save_path + ".report", 'a') as outfile:
            outfile.write(log_str)
        return log_str, reward_mean

    def get_report(self):
        episode_report = [self.facades[i].get_episode_report(10) for i in range(self.population_size)]
        train_report = {k: np.mean(v[-10:]) for k, v in self.training_history.items()}
        reward_mean = np.mean([report['average_total_reward'] for report in episode_report])
        return episode_report, train_report, reward_mean

    def test_report(self, a, b, prefix=""):
        with open(self.save_path + prefix + ".test_rewards.npy", 'wb') as outfile:
            np.save(outfile, a)
        with open(self.save_path + prefix + ".test_choices.npy", 'wb') as outfile:
            np.save(outfile, self.choice_history)
        with open(self.save_path + prefix + ".test_steps.npy", 'wb') as outfile:
            np.save(outfile, b)
        print('write test report finished')

    def compute_actor_loss(self, agent, observation, distr_taus_risk):
        """ Compute CVaR of the reward distribution given state and action
        selected by risk-averse policy
        """
        # action = self.actor(state)
        policy_output = self.buffer_facade.apply_policy(observation, agent)
        tau_actor_k = distr_taus_risk.sample((self.K,)).to(self.device)
        tail_samples = self.critic.get_sampled_Z(
            observation, tau_actor_k, policy_output)  # [batch_size x K]

        cvar = tail_samples.mean()
        return cvar

    def compute_Z_value(self, agent, observation, distr_taus_risk):
        """ Compute CVaR of the reward distribution given state and action
        selected by risk-averse policy
        """
        # action = self.actor(state)
        policy_output = self.buffer_facade.apply_policy(observation, agent)
        tau_actor_k = distr_taus_risk.sample((2*self.K,)).to(self.device)
        tail_samples = self.critic.get_sampled_Z(
            observation, tau_actor_k, policy_output)  # [batch_size x K]

        cvar = tail_samples.mean(dim=1)
        return cvar

    def compute_Z_value_with_tau(self, agent, observation, tau_actor_k):
        """ Compute CVaR of the reward distribution given state and action
        selected by risk-averse policy
        """
        # action = self.actor(state)
        policy_output = self.buffer_facade.apply_policy(observation, agent)
        tail_samples = self.critic.get_sampled_Z(
            observation, tau_actor_k, policy_output)  # [batch_size x K]

        cvar = tail_samples.mean(dim=1)
        return cvar

    def quantile_huber_loss(self, T_theta, Theta, tau_quantiles, k=1):
        """Compute quantile huber loss.

        Parameters
        ----------
        T_theta: torch.Tensor
                Target quantiles of size [batch_size x num_quantiles]

        Theta: torch.Tensor
                Current quantiles of size [batch_size x num_quantiles]
        tau_quantiles: torch.Tensor
            Quantile levles: [1xnum_quantiles]

        Returns
        -------
        loss: tensor
            Quantile Huber loss
        """
        # Repeat Theta rows N times, amd stack batches in 3dim -->
        # -->[batch_size x N x N ]
        # (N = num quantiles)
        # Repeat T_Theta cols N times, amd stack batches in 3dim -->
        # --> [batch_size x N x N ]

        batch_size, num_quantiles = Theta.size()
        Theta_ = Theta.unsqueeze(2)  # batch_size, N, 1
        T_theta_ = T_theta.unsqueeze(1)  # batch_size, 1, N
        tau = tau_quantiles.unsqueeze(0).unsqueeze(2)  # 1, N,1
        error = T_theta_ - Theta_  # all minus all [batch_size, N, N]

        quantile_loss = torch.abs(tau - error.le(0.).float())  # (batch_size, N, N)

        huber_loss_ = F.smooth_l1_loss(
            Theta_.expand(-1, -1, num_quantiles),
            T_theta_.expand(-1, num_quantiles, -1),
            reduction='none')

        loss_ = (quantile_loss * huber_loss_).mean()
        return loss_

    def td(self, observations, policy_outputs, rewards, next_observations, done_masks):
        all_observations = utils.torch_cat_dict(observations)
        all_policy_outputs = utils.torch_cat_dict(policy_outputs)
        all_rewards = torch.cat(rewards)
        all_next_observations = utils.torch_cat_dict(next_observations)
        all_done_masks = torch.cat(done_masks)
        tau_k = self.distr_taus_uniform.sample(
            (self.N,)).to(self.device)  # [batch_size x N]
        tau_k_ = self.distr_taus_uniform.sample(
            (self.N,)).to(self.device)  # [batch_size x N]

        # [batch_size x num_confidences]
        Z_tau_K = self.critic.get_sampled_Z(all_observations, tau_k, all_policy_outputs)
        with torch.no_grad():
            next_policy_output = utils.torch_cat_dict([self.buffer_facade.apply_policy(next_observation, agent)
                                                       for agent, next_observation in
                                                       zip(self.population, next_observations)])
            Z_next_tau_K = self.critic_target.get_sampled_Z(
                all_next_observations, tau_k_, next_policy_output)

            all_done_masks = all_done_masks.unsqueeze(-1).expand_as(Z_next_tau_K)
            all_rewards = all_rewards.unsqueeze(-1).expand_as(Z_next_tau_K)
            target_Z_tau_K = all_rewards + self.gamma * Z_next_tau_K * (1 - all_done_masks.int())

        return Z_tau_K, target_Z_tau_K, tau_k

    def update_distr_taus_risk(self, i):
        self.cur_alpha_cvar_list = [self.cal_decay_value(alpha, i, self.n_iter[-1]) for alpha in self.alpha_cvar_list]
        self.distr_taus_risks = [uniform.Uniform(b_ac, ac)
                                 for b_ac, ac in zip(self.below_alpha_cvar_list, self.cur_alpha_cvar_list)]

    def save(self):
        torch.save(self.critic.state_dict(), self.save_path + "_critic")
        torch.save(self.critic_optimizer.state_dict(), self.save_path + "_critic_optimizer")
        torch.save(self.population_optimizer.state_dict(), self.save_path + "_actor_optimizer")
        for i, actor in enumerate(self.population):
            torch.save(actor.state_dict(), self.save_path + "_actor_" + str(i))

        print("save finished")

    def load(self):
        self.critic.load_state_dict(torch.load(self.save_path + "_critic", map_location=self.device))
        self.critic_optimizer.load_state_dict(
            torch.load(self.save_path + "_critic_optimizer", map_location=self.device))
        self.critic_target = copy.deepcopy(self.critic)
        self.population_optimizer.load_state_dict(
            torch.load(self.save_path + "_actor_optimizer", map_location=self.device))

        for i, actor in enumerate(self.population):
            actor.load_state_dict(torch.load(self.save_path + "_actor_" + str(i), map_location=self.device))
            self.population_target[i] = copy.deepcopy(actor)

    def run_an_episode(self, epsilon, cvar=1, initial_observation=None, with_train=False, pick_rows=None):
        '''
        Run episode for a batch of user
        @input:
        - epsilon: greedy epsilon for random exploration
        - initial_observation
        - with_train: apply batch training for each step of the episode
        - pick_rows: pick certain rows of the data when reseting the environment
        '''
        # observation --> state, action
        cvar_taus_uniform = uniform.Uniform(0, cvar)
        if initial_observation:
            observation = initial_observation
        elif pick_rows:
            observation = self.buffer_facade.reset_env({"batch_size": len(pick_rows), 'pick_rows': pick_rows})
        else:
            observation = self.buffer_facade.reset_env({"batch_size": self.episode_batch_size})
        step = 0
        done = [False] * self.batch_size
        train_report = None
        choice_history = []
        while sum(done) < len(done):
            step += 1
            with torch.no_grad():
                # sample action
                agent_idx, max_Z_value = np.array([None for _ in range(self.batch_size)]), np.array([float('-inf') for _ in range(self.batch_size)])
                # for i, (agent, distr_taus_risk) in enumerate(zip(self.population, self.distr_taus_risks)):
                # action_policy = {"action": torch.Tensor([None for _ in range(self.batch_size)]).to(self.device),
                #                  "action_features": torch.Tensor([None for _ in range(self.batch_size)]).to(self.device)}
                action_policy = {}
                tau_actor_k = cvar_taus_uniform.sample((2 * self.K,)).to(self.device)
                for i, agent in enumerate(self.population):
                    Z_value = self.compute_Z_value_with_tau(agent, observation, tau_actor_k)
                    Z_value = Z_value.cpu().detach().numpy()
                    # Q_value = self.compute_actor_loss(agent, observation, distr_taus_risk)
                    idx = np.where(max_Z_value < Z_value)
                    agent_idx[idx] = i
                    max_Z_value[idx] = Z_value[idx]
                    policy_output = self.buffer_facade.apply_policy(observation, self.population[i], epsilon, do_explore=False)
                    if action_policy:
                        action_policy["action"][idx] = policy_output["action"][idx].clone().detach()
                        action_policy["action_emb"][idx] = policy_output["action_emb"][idx].clone().detach()
                        action_policy["action_features"][idx] = policy_output["action_features"][idx].clone().detach()
                    else:
                        action_policy["action"] = policy_output["action"].clone().detach()
                        action_policy["action_emb"] = policy_output["action_emb"].clone().detach()
                        action_policy["action_features"] = policy_output["action_features"].clone().detach()
                choice_history.append(agent_idx)
                # policy_output = self.facade.apply_policy(observation, self.actor, epsilon, do_explore=False)
                # apply action on environment and update replay buffer
                # next_observation, reward, done, info = self.facade.env_step(policy_output)
                next_observation, reward, done, info, ret = self.buffer_facade.env_step_without_new_sample(action_policy)
                # update replay buffer
                if not pick_rows:
                    self.buffer_facade.update_buffer(observation, policy_output, reward, done, next_observation, info)
                # observate for the next step
                observation = next_observation
            if with_train:
                train_report = self.step_train()
        choice_history = np.transpose(choice_history)
        choice_history = np.pad(choice_history, ((0, 0), (0, 20-choice_history.shape[1])), mode='constant', constant_values=0)
        self.choice_history.extend(choice_history)
        return ret

    def train(self):
        if len(self.n_iter) > 2:
            self.load()

        print("Run procedures before training")
        self.action_before_train()
        bandit = BetaBernoulliBandit()
        t = time.time()
        start_time = t
        # training
        print("Training:")
        observations = [self.facades[i].reset_env({"batch_size": self.episode_batch_size})
                        for i in range(self.population_size)]
        step_offset = sum(self.n_iter[:-1])
        for i in tqdm(range(step_offset, step_offset + self.n_iter[-1]), ncols=50):
            observations = self.run_episode_step(i, self.exploration_scheduler.value(i), observations, True)
            if i % self.train_every_n_step == 0:
                self.step_train()
                self.update_distr_taus_risk(i)
            if i % self.check_episode == 0:
                t_ = time.time()
                print(f"Episode step {i}, time diff {t_ - t}, total time dif {t - start_time})")
                log_str, reward_mean = self.log_iteration(i)
                print(log_str)
                bandit.update_dist(reward_mean)
                self.diversity_importance = bandit.sample()
                t = t_
                if i % (3 * self.check_episode) == 0:
                    self.save()
        self.action_after_train()

    def cal_decay_value(self, alpha, m, M):
        return max(alpha, 1 - (1 - alpha) * m / (self.risk_factor_rou * M))

    def set_eval(self):
        for actor in self.population:
            actor.eval()

    def test(self, cvar=1, test_sim=False):
        # Testing
        for facade in self.facades:
            facade.initialize_train()
        self.load()
        self.set_eval()
        self.critic.eval()
        print("Testing:")
        # for i in tqdm(range(self.n_iter)):
        reward_history, step_history = [], []
        with torch.no_grad():
            for i in tqdm(range(len(self.buffer_facade.env.reader) // self.batch_size), ncols=50):
                pick_rows = [row for row in range(i * self.batch_size, (i + 1) * self.batch_size)]
                ret = self.run_an_episode(self.exploration_scheduler.value(i), cvar=cvar, pick_rows=pick_rows)
                #ret = self.plot_tsne(self.exploration_scheduler.value(i), cvar=cvar, pick_rows=pick_rows)
                reward_history.extend(ret[0])
                step_history.extend(ret[1])

        self.test_report(reward_history, step_history, "_critic_cvar" + str(cvar) + "_test_sim" if test_sim else "")

    def step_train(self):
        # observation, policy_output, reward, done_mask, next_observation = self.buffer_facade.sample_buffer(self.batch_size)
        #         reward = torch.FloatTensor(reward)
        #         done_mask = torch.FloatTensor(done_mask)
        # buffer_items = [self.buffer_facade.sample_buffer(self.batch_size) for i in range(self.population_size)]
        observations, policy_outputs, rewards, done_masks, next_observations = \
            list(zip(*[self.buffer_facade.sample_buffer(self.batch_size) for _ in range(len(self.population))]))

        critic_loss, actor_loss, div_loss, behavior_loss = self.get_loss(observations, policy_outputs, rewards, done_masks, next_observations)
        self.training_history['actor_loss'].append(actor_loss.item())
        self.training_history['critic_loss'].append(critic_loss.item())
        self.training_history['diversity_loss'].append(div_loss.item())
        self.training_history['behavior_loss'].append(behavior_loss.item())

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for actor, actor_target in zip(self.population, self.population_target):
            for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {"step_loss": (self.training_history['actor_loss'][-1],
                              self.training_history['critic_loss'][-1],
                              self.training_history['diversity_loss'][-1],
                              self.training_history['behavior_loss'][-1])}

    def cal_behavior_loss(self, observations, policy_outputs, next_observations, do_update=True):
        #         observation, exposure, feedback = self.facade.sample_supervise_data(self.batch_size)
        behavior_losses = 0
        for observation, policy_output, next_observation, actor in \
                zip(observations, policy_outputs, next_observations, self.population):
            observation, exposure, feedback = self.buffer_facade.extract_behavior_data(observation, policy_output,
                                                                                next_observation)
            observation['candidate_ids'] = exposure['ids']
            observation['candidate_features'] = exposure['features']
            policy_output = self.buffer_facade.apply_policy(observation, actor, do_softmax=False)
            action_prob = torch.sigmoid(policy_output['candidate_prob'])
            behavior_loss = F.binary_cross_entropy(action_prob, feedback)
            behavior_losses += behavior_loss
        return behavior_losses

    def get_loss(self, observations, policy_outputs, rewards, done_masks, next_observations,
                 do_actor_update=True, do_critic_update=True):

        # Critic Training:
        current_Z, target_Z, tau_k = self.td(
            observations, policy_outputs, rewards, next_observations, done_masks)
        # update critic network
        critic_loss = self.quantile_huber_loss(target_Z, current_Z, tau_k)
        # Regularization loss
        #         critic_reg = current_critic_output['reg']

        if do_critic_update and self.critic_lr > 0:
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        # Compute actor loss
        actor_losses = 0
        for agent, observation, distr_taus_risk in zip(self.population, observations, self.distr_taus_risks):
            actor_loss = -self.compute_actor_loss(agent, observation, distr_taus_risk)
            actor_losses = actor_losses + actor_loss

        observation = self.buffer_facade.sample_buffer(self.embedding_size)[0]
        diversity_loss = self.compute_population_loss(observation)
        behavior_loss = self.cal_behavior_loss(observations, policy_outputs, next_observations)
        actor_losses += self.reg_weight * (self.diversity_importance * diversity_loss + (0.5-self.diversity_importance) * behavior_loss)

        if do_actor_update and self.actor_lr > 0:
            # Optimize the actor
            self.population_optimizer.zero_grad()
            actor_losses.backward()
            self.population_optimizer.step()

        return critic_loss, actor_losses, diversity_loss, behavior_loss
