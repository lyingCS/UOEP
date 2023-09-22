import time
import copy
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import utils
from model.agents.BaseRLAgent import BaseRLAgent


class TD3(BaseRLAgent):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - args from DDPG:
            - episode_batch_size
            - batch_size
            - actor_lr
            - critic_lr
            - actor_decay
            - critic_decay
            - target_mitigate_coef
            - args from BaseRLAgent:
                - gamma
                - n_iter
                - train_every_n_step
                - initial_greedy_epsilon
                - final_greedy_epsilon
                - elbow_greedy
                - check_episode
                - with_eval
                - save_path
        '''
        parser = BaseRLAgent.parse_model_args(parser)
        parser.add_argument('--episode_batch_size', type=int, default=8,
                            help='episode sample batch size')
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

        return parser

    def __init__(self, args, facade):
        '''
        self.gamma
        self.n_iter
        self.check_episode
        self.with_eval
        self.save_path
        self.facade
        self.exploration_scheduler
        '''
        super().__init__(args, facade)
        self.episode_batch_size = args.episode_batch_size
        self.batch_size = args.batch_size

        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.actor_decay = args.actor_decay
        self.critic_decay = args.critic_decay

        self.actor = facade.actor
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr,
                                                weight_decay=args.actor_decay)
        self.critic = facade.critic
        self.critic1 = self.critic[0]
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=args.critic_lr,
                                                  weight_decay=args.critic_decay)

        self.critic2 = self.critic[1]
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=args.critic_lr,
                                                  weight_decay=args.critic_decay)

        self.tau = args.target_mitigate_coef
        if len(self.n_iter) == 1:
            with open(self.save_path + ".report", 'w') as outfile:
                outfile.write(f"{args}\n")

    # def action_before_train(self):
    #     '''
    #     Action before training:
    #     - facade setup:
    #         - buffer setup
    #     - run random episodes to build-up the initial buffer
    #     '''
    #     self.facade.initialize_train() # buffer setup
    #     prepare_step = 0
    #     # random explore before training
    #     initial_epsilon = 1.0
    #     observation = self.facade.reset_env({"batch_size": self.episode_batch_size})
    #     while not self.facade.is_training_available:
    #         observation = self.run_episode_step(0, initial_epsilon, observation, True)
    #         prepare_step += 1
    #     # training records
    #     self.training_history = {"critic1_loss": [], "critic2_loss": [], "actor_loss": []}

    #     print(f"Total {prepare_step} prepare steps")
    def action_before_train(self):
        '''
        Action before training:
        - facade setup:
            - buffer setup
        - run random episodes to build-up the initial buffer
        '''
        self.facade.initialize_train()  # buffer setup
        prepare_step = 0
        # random explore before training
        initial_epsilon = 1.0
        observation = self.facade.reset_env({"batch_size": self.episode_batch_size})
        while not self.facade.is_training_available:
            observation = self.run_episode_step(0, initial_epsilon, observation, True)
            prepare_step += 1
        # training records
        self.training_history = {'actor_loss': [], 'critic1_loss': [], 'critic2_loss': []}

        print(f"Total {prepare_step} prepare steps")

    def run_episode_step(self, *episode_args):
        '''
        One step of interaction
        '''
        episode_iter, epsilon, observation, do_buffer_update = episode_args
        with torch.no_grad():
            # sample action
            # policy_output = self.facade.apply_policy(observation, self.actor, epsilon, do_explore=False)
            policy_output = self.facade.apply_policy(observation, self.actor, epsilon, do_explore=True)
            # apply action on environment and update replay buffer
            next_observation, reward, done, info = self.facade.env_step(policy_output)
            # update replay buffer
            if do_buffer_update:
                self.facade.update_buffer(observation, policy_output, reward, done, next_observation, info)
        return next_observation

    def step_train(self):
        observation, policy_output, reward, done_mask, next_observation = self.facade.sample_buffer(self.batch_size)
        # reward = torch.FloatTensor(reward)
        # done_mask = torch.FloatTensor(done_mask)

        critic_loss, actor_loss = self.get_td3_loss(observation, policy_output, reward, done_mask, next_observation)
        self.training_history['actor_loss'].append(actor_loss.item())
        self.training_history['critic1_loss'].append(critic_loss[0])
        self.training_history['critic2_loss'].append(critic_loss[1])

        # Update the frozen target models
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {"step_loss": (self.training_history['actor_loss'][-1],
                              self.training_history['critic1_loss'][-1],
                              self.training_history['critic2_loss'][-1])}

    def get_td3_loss(self, observation, policy_output, reward, done_mask, next_observation,
                     do_actor_update=True, do_critic_update=True):
        '''
        @input:
        - observation: {'user_profile': {'user_id': (B,),
                                         'uf_{feature_name}': (B, feature_dim)},
                        'user_history': {'history': (B, max_H),
                                         'history_if_{feature_name}': (B, max_H, feature_dim),
                                         'history_{response}': (B, max_H),
                                         'history_length': (B, )}}
        - policy_output: {'state': (B, state_dim),
                          'action: (B, action_dim)}
        - reward: (B,)
        - done_mask: (B,)
        - next_observation: the same format as @input-observation
        '''

        # Compute the target Q value
        next_policy_output = self.facade.apply_policy(next_observation, self.actor_target)
        target_critic1_output = self.facade.apply_critic(next_observation, next_policy_output, self.critic1_target)
        target_critic2_output = self.facade.apply_critic(next_observation, next_policy_output, self.critic2_target)
        target_Q = torch.min(target_critic1_output['q'], target_critic2_output['q'])
        # r+gamma*Q' when done; r+Q when not done
        # target_Q = reward + ((self.gamma * done_mask) + (1 - done_mask)) * target_Q.detach()
        # target_Q = reward + ((self.gamma * done_mask) + torch.logical_not(done_mask)) * target_Q.detach()
        target_Q = reward + self.gamma * ((1 - done_mask.int()) * target_Q).detach()

        critic_loss_list = []
        if do_critic_update and self.critic_lr > 0:
            for critic, optimizer in [(self.critic1, self.critic1_optimizer),
                                      (self.critic2, self.critic2_optimizer)]:
                # Get current Q estimate
                current_critic_output = self.facade.apply_critic(observation,
                                                                 utils.wrap_batch(policy_output, device=self.device),
                                                                 critic)
                current_Q = current_critic_output['q']
                # Compute critic loss
                critic_loss = F.mse_loss(current_Q, target_Q).mean()
                critic_loss_list.append(critic_loss.item())

                # Optimize the critic
                optimizer.zero_grad()
                critic_loss.backward()
                optimizer.step()

        # Compute actor loss
        policy_output = self.facade.apply_policy(observation, self.actor)
        critic_output = self.facade.apply_critic(observation, policy_output, self.critic1)
        actor_loss = -critic_output['q'].mean()

        if do_actor_update and self.actor_lr > 0:
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        return critic_loss_list, actor_loss

    def save(self):
        torch.save(self.critic1.state_dict(), self.save_path + "_critic1")
        torch.save(self.critic1_optimizer.state_dict(), self.save_path + "_critic1_optimizer")

        torch.save(self.critic2.state_dict(), self.save_path + "_critic2")
        torch.save(self.critic2_optimizer.state_dict(), self.save_path + "_critic2_optimizer")

        torch.save(self.actor.state_dict(), self.save_path + "_actor")
        torch.save(self.actor_optimizer.state_dict(), self.save_path + "_actor_optimizer")

    def load(self):
        self.critic1.load_state_dict(torch.load(self.save_path + "_critic1", map_location=self.device))
        self.critic1_optimizer.load_state_dict(
            torch.load(self.save_path + "_critic1_optimizer", map_location=self.device))
        self.critic1_target = copy.deepcopy(self.critic1)

        self.critic2.load_state_dict(torch.load(self.save_path + "_critic2", map_location=self.device))
        self.critic2_optimizer.load_state_dict(
            torch.load(self.save_path + "_critic2_optimizer", map_location=self.device))
        self.critic2_target = copy.deepcopy(self.critic2)

        self.actor.load_state_dict(torch.load(self.save_path + "_actor", map_location=self.device))
        self.actor_optimizer.load_state_dict(torch.load(self.save_path + "_actor_optimizer", map_location=self.device))
        self.actor_target = copy.deepcopy(self.actor)

    def test(self, test_sim=False):
        t = time.time()
        # Testing
        self.facade.initialize_train()
        self.load()
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
        print("Testing:")
        # for i in tqdm(range(self.n_iter)):
        reward_history, step_history = [], []
        with torch.no_grad():
            for i in tqdm(range(len(self.facade.env.reader) // self.batch_size), ncols=50):
                pick_rows = [row for row in range(i * self.batch_size, (i + 1) * self.batch_size)]
                ret = self.run_an_episode(self.exploration_scheduler.value(i), pick_rows=pick_rows)
                reward_history.extend(ret[0])
                step_history.extend(ret[1])

        self.test_report(reward_history, step_history, "_test_sim" if test_sim else "")

    def test_report(self, a, b, prefix=""):
        with open(self.save_path + prefix + ".test_rewards.npy", 'wb') as outfile:
            np.save(outfile, a)
        with open(self.save_path + prefix + ".test_steps.npy", 'wb') as outfile:
            np.save(outfile, b)
        print('write test report finished')

    def run_an_episode(self, epsilon, initial_observation=None, with_train=False, pick_rows=None):
        '''
        Run episode for a batch of user
        @input:
        - epsilon: greedy epsilon for random exploration
        - initial_observation
        - with_train: apply batch training for each step of the episode
        - pick_rows: pick certain rows of the data when reseting the environment
        '''
        # observation --> state, action
        if initial_observation:
            observation = initial_observation
        elif pick_rows:
            observation = self.facade.reset_env({"batch_size": len(pick_rows), 'pick_rows': pick_rows})
        else:
            observation = self.facade.reset_env({"batch_size": self.episode_batch_size})
        step = 0
        done = [False] * self.batch_size
        train_report = None
        while sum(done) < len(done):
            step += 1
            with torch.no_grad():
                # sample action
                policy_output = self.facade.apply_policy(observation, self.actor, epsilon, do_explore=True)
                # policy_output = self.facade.apply_policy(observation, self.actor, epsilon, do_explore=False)
                # apply action on environment and update replay buffer
                # next_observation, reward, done, info = self.facade.env_step(policy_output)
                next_observation, reward, done, info, ret = self.facade.env_step_without_new_sample(policy_output)
                # update replay buffer
                if not pick_rows:
                    self.facade.update_buffer(observation, policy_output, reward, done, next_observation, info)
                # observate for the next step
                observation = next_observation
            if with_train:
                train_report = self.step_train()
        return ret
