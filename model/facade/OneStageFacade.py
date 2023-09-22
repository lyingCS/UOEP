import torch
import torch.nn.functional as F
import random
import numpy as np
import utils


class OneStageFacade():
    '''
    The general interface for one-stage RL policies.
    Key components:
    - replay buffer
    - environment
    - actor
    - critic
    '''

    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - slate_size
        - buffer_size
        - start_timestamp
        - noise_var
        - q_laplace_smoothness
        - topk_rate
        - empty_start_rate
        '''
        parser.add_argument('--slate_size', type=int, default=6,
                            help='slate size for actions')
        parser.add_argument('--buffer_size', type=int, default=10000,
                            help='replay buffer size')
        parser.add_argument('--start_timestamp', type=int, default=1000,
                            help='start timestamp for buffer sampling')
        parser.add_argument('--noise_var', type=float, default=0,
                            help='noise magnitude for action embedding sampling')
        parser.add_argument('--q_laplace_smoothness', type=float, default=0.5,
                            help='critic smoothness scalar for actors')
        parser.add_argument('--topk_rate', type=float, default=1.,
                            help='rate choosing topk rather than categorical sampling for items')
        parser.add_argument('--empty_start_rate', type=float, default=0,
                            help='probability of starting an episode from empty history')
        return parser

    def __init__(self, args, environment, actor, critic):
        super().__init__()
        self.device = args.device
        self.env = environment
        self.actor = actor
        self.critic = critic

        self.slate_size = args.slate_size
        self.noise_var = args.noise_var
        self.noise_decay = args.noise_var / args.n_iter[-1]
        self.q_laplace_smoothness = args.q_laplace_smoothness
        self.topk_rate = args.topk_rate
        self.empty_start_rate = args.empty_start_rate

        self.n_item = self.env.action_space['item_id'][1]
        # (N,)
        self.candidate_iids = np.arange(1, self.n_item + 1)
        # (N, item_dim)
        self.candidate_features = torch.FloatTensor(self.env.reader.get_item_list_meta(self.candidate_iids)).to(
            self.device)
        self.candidate_iids = torch.tensor(self.candidate_iids).to(self.device)

        # replay buffer is initialized in initialize_train()
        self.buffer_size = args.buffer_size
        self.start_timestamp = args.start_timestamp

    def initialize_train(self):
        '''
        Procedures before training
        '''
        # replay buffer
        self.buffer = {
            "user_profile": torch.zeros(self.buffer_size, self.env.reader.portrait_len),
            "history": torch.zeros(self.buffer_size, self.env.reader.max_seq_len).to(torch.long),
            # "history_features": torch.zeros(self.buffer_size, self.env.reader.max_seq_len,
            # self.env.reader.item_vec_size),
            "next_history": torch.zeros(self.buffer_size, self.env.reader.max_seq_len).to(torch.long),
            # "next_history_features": torch.zeros(self.buffer_size, self.env.reader.max_seq_len,
            # self.env.reader.item_vec_size),
            "state_emb": torch.zeros(self.buffer_size, self.actor.state_dim),
            "action_emb": torch.zeros(self.buffer_size, self.actor.action_dim),
            # item filtering network as hyper-action
            "action": torch.zeros(self.buffer_size, self.slate_size, dtype=torch.long),  # item slate as actual action
            "reward": torch.zeros(self.buffer_size),
            "feedback": torch.zeros(self.buffer_size, self.slate_size),
            "done": torch.zeros(self.buffer_size, dtype=torch.bool)
        }
        for k, v in self.buffer.items():
            self.buffer[k] = v.to(self.device)
        self.buffer_head = 0
        self.current_buffer_size = 0
        self.n_stream_record = 0
        self.is_training_available = False

    def reset_env(self, initial_params={"batch_size": 1}):
        '''
        Reset user response environment
        '''
        # Note: avoid starting from empty history in order to get more positive feedback immediately.
        # This would speed up the training but make limit the initial eploration.
        initial_params['empty_history'] = True if np.random.rand() < self.empty_start_rate else False
        initial_observation = self.env.reset(initial_params)
        return initial_observation

    def env_step_without_new_sample(self, policy_output):
        action_dict = {'action': policy_output['action'],
                       'action_features': policy_output['action_features']}
        observation, reward, done, info, ret = self.env.step_without_new_sample(action_dict)
        return observation, reward, done, info, ret

    def env_step(self, policy_output):
        action_dict = {'action': policy_output['action'],
                       'action_features': policy_output['action_features']}
        observation, reward, done, info = self.env.step(action_dict)
        return observation, reward, done, info

    def stop_env(self):
        self.env.stop()

    def get_episode_report(self, n_recent=10):
        recent_rewards = self.env.reward_history[-10:]
        recent_steps = self.env.step_history[-10:]
        episode_report = {'average_total_reward': np.mean(recent_rewards),
                          'reward_variance': np.var(recent_rewards),
                          'max_total_reward': np.max(recent_rewards),
                          'min_total_reward': np.min(recent_rewards),
                          'average_n_step': np.mean(recent_steps),
                          'max_n_step': np.max(recent_steps),
                          'min_n_step': np.min(recent_steps),
                          'buffer_size': self.current_buffer_size}
        return episode_report

    def apply_critic(self, observation, policy_output, critic_model):
        feed_dict = {"state_emb": policy_output["state_emb"],
                     "action_emb": policy_output["action_emb"]}
        critic_output = critic_model(feed_dict)
        return critic_output

    #     def get_critic_scalar(self, critic_output):
    #         new_q = torch.abs(critic_output['q'].detach()) # (B,)
    #         for q in new_q.numpy():
    #             if len(self.hist_Q) < self.start_timestamp:
    #                 self.hist_Q.append(q)
    #             elif np.random.rand() < (1.0 / (self.n_stream_record + 1)):
    #                 self.hist_Q[np.random.randint(0,self.start_timestamp)] = q
    #         return (np.mean(self.hist_Q) + self.q_laplace_smoothness) / (new_q + self.q_laplace_smoothness)

    def apply_policy(self, observation, policy_model, epsilon=0,
                     do_explore=False, do_softmax=True):
        '''
        @input:
        - observation: input of policy model
        - policy_model
        - epsilon: greedy epsilon, effective only when do_explore == True
        - do_explore: exploration flag, True if adding noise to action
        - do_softmax: output softmax score
        '''
        #         feed_dict = utils.wrap_batch(observation, device = self.device)
        feed_dict = observation
        out_dict = policy_model(feed_dict)
        if do_explore:
            action_emb = out_dict['action_emb']
            # sampling noise of action embedding
            if np.random.rand() < epsilon:
                # action_emb = torch.clamp(torch.rand_like(action_emb) * self.noise_var, -1, 1)
                action_emb = torch.clamp(torch.randn_like(action_emb) * self.noise_var, -1, 1)
            else:
                # action_emb = action_emb + torch.clamp(torch.rand_like(action_emb) * self.noise_var, -1, 1)
                action_emb = action_emb + torch.clamp(torch.randn_like(action_emb) * self.noise_var, -1, 1)
            #                 self.noise_var -= self.noise_decay
            out_dict['action_emb'] = action_emb

        if 'candidate_ids' in feed_dict:
            # (B, L, item_dim)
            out_dict['candidate_features'] = feed_dict['candidate_features']
            # (B, L)
            out_dict['candidate_ids'] = feed_dict['candidate_ids']
            batch_wise = True
        else:
            # (1,L,item_dim)
            out_dict['candidate_features'] = self.candidate_features.unsqueeze(0)
            # (L,)
            out_dict['candidate_ids'] = self.candidate_iids
            batch_wise = False

        # action prob (B,L)
        action_prob = policy_model.score(out_dict['action_emb'],
                                         out_dict['candidate_features'],
                                         do_softmax=do_softmax)

        # two types of greedy selection
        if np.random.rand() >= self.topk_rate:
            # greedy random: categorical sampling
            action, indices = utils.sample_categorical_action(action_prob, out_dict['candidate_ids'],
                                                              self.slate_size, with_replacement=False,
                                                              batch_wise=batch_wise, return_idx=True)
        else:
            # indices on action_prob
            _, indices = torch.topk(action_prob, k=self.slate_size, dim=1)
            # topk action
            if batch_wise:
                action = torch.gather(out_dict['candidate_ids'], 1, indices).detach()  # (B, slate_size)
            else:
                action = out_dict['candidate_ids'][indices].detach()  # (B, slate_size)
        # (B,K)
        out_dict['action'] = action
        # (B,K,item_dim)
        out_dict['action_features'] = self.candidate_features[action - 1]
        # (B,K)
        out_dict['action_prob'] = torch.gather(action_prob, 1, indices)
        # (B,L)
        out_dict['candidate_prob'] = action_prob
        return out_dict

    def sample_buffer(self, batch_size):
        '''
        Batch sample is organized as a tuple of (observation, policy_output, reward, next_observation)
        '''
        indices = np.random.randint(0, self.current_buffer_size, size=batch_size)
        U, H, N, S, HA, A, R, F, D = self.read_buffer(indices)
        observation = {"user_profile": U, "history_features": H}
        policy_output = {"state_emb": S, "action_emb": HA, "action": A}
        reward = R
        done_mask = D
        next_observation = {"user_profile": U, "history_features": N, "previous_feedback": F}
        return observation, policy_output, reward, done_mask, next_observation

    def sample_raw_data(self, batch_size):
        '''
        Sample supervise data from raw training data
        @output:
        - observation: {"user_profile": tensor , "history_features": tensor}
        - exposure: {"ids": tensor, "features": tensor}
        - user_feedback: tensor
        '''
        batch = self.env.sample_user(batch_size, with_feedback=True)
        observation = {"user_profile": batch["user_profile"], "history_features": batch["history_features"]}
        exposure = {"ids": batch["exposed_items"], "features": batch["exposed_item_features"]}
        user_feedback = torch.FloatTensor(batch["feedback"])
        return observation, exposure, user_feedback

    def extract_behavior_data(self, observation, policy_output, next_observation):
        '''
        Extract supervise data from RL samples (observation, policy_output, next_observation)
        @output:
        - observation: {"user_profile": tensor , "history_features": tensor}
        - exposure: {"ids": tensor, "features": tensor}
        - user_feedback: tensor
        '''
        observation = {"user_profile": observation["user_profile"],
                       "history_features": observation["history_features"]}
        exposed_items = policy_output["action"]
        exposure = {"ids": exposed_items,
                    "features": self.candidate_features[exposed_items - 1]}
        user_feedback = next_observation["previous_feedback"]
        return observation, exposure, user_feedback

    #     def get_policy_gradient_loss(self, observation, policy_output, next_observation, actor):
    #         observation, exposure, feedback = self.extract_supervise_data(observation, policy_output, next_observation)
    #         policy_output = self.apply_policy(observation, actor, candidates = exposure, do_softmax = False)
    #         action_prob = torch.sigmoid(policy_output['action_prob'])
    #         behavior_loss = F.binary_cross_entropy(action_prob, feedback)
    #         return behavior_loss

    def update_buffer(self, observation, policy_output, reward, done_mask, next_observation, info):
        '''
        Each sample is organized as a tuple of (U,H,N,S,HA,A,R,F,D):
        - (U)ser profile: vector
        - (H)istory: tensor
        - (N)ext history: tensor
        - (S)tate embedding: vector 
        - (H)yper-(A)ction embedding: vector
        - (A)ction: list of ids
        - (R)eward: scalar
        - (F)eedback: user feedback for A
        - (D)one: done flag
        '''
        if self.buffer_head + reward.shape[0] >= self.buffer_size:
            tail = self.buffer_size - self.buffer_head
            indices = [self.buffer_head + i for i in range(tail)] + \
                      [i for i in range(reward.shape[0] - tail)]
        else:
            indices = [self.buffer_head + i for i in range(reward.shape[0])]

        # update buffer
        self.buffer["user_profile"][indices] = observation['user_profile']
        self.buffer["history"][indices] = observation['history']
        self.buffer["next_history"][indices] = next_observation['history']
        self.buffer["state_emb"][indices] = policy_output['state_emb']
        self.buffer["action"][indices] = policy_output['action']
        self.buffer["action_emb"][indices] = policy_output['action_emb']
        self.buffer["reward"][indices] = reward
        self.buffer["feedback"][indices] = info['response']
        self.buffer["done"][indices] = done_mask
        # buffer pointer
        self.buffer_head = (self.buffer_head + reward.shape[0]) % self.buffer_size
        self.n_stream_record += reward.shape[0]
        self.current_buffer_size = min(self.n_stream_record, self.buffer_size)
        # training is available when sufficient sample bufferred
        if self.n_stream_record >= self.start_timestamp:
            self.is_training_available = True

    def read_buffer(self, indices):
        U = self.buffer["user_profile"][indices]
        H = self.candidate_features[self.buffer["history"][indices] - 1]
        #         H = self.buffer["history_features"][indices]
        N = self.candidate_features[self.buffer["next_history"][indices] - 1]
        #         N = self.buffer["next_history_features"][indices]
        S = self.buffer["state_emb"][indices]
        HA = self.buffer["action_emb"][indices]
        A = self.buffer["action"][indices]
        R = self.buffer["reward"][indices]
        F = self.buffer["feedback"][indices]
        D = self.buffer["done"][indices]
        return U, H, N, S, HA, A, R, F, D
