import torch
import torch.nn.functional as F
import random
import numpy as np
import utils

from model.facade.OneStageFacade import OneStageFacade


class TwoStageFacade(OneStageFacade):
    
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - candidate_size
        - stage2_noise_var
        - args from OneStageFacade:
            - slate_size
            - buffer_size
            - start_timestamp
            - noise_var
            - q_laplace_smoothness
            - topk_rate
            - empty_start_rate
        '''
        parser = OneStageFacade.parse_model_args(parser)
        parser.add_argument('--candidate_size', type=int, default=100, 
                            help='intermediate candidate item set size')
        parser.add_argument('--stage2_noise_var', type=float, default=0.1, 
                            help='noise magnitude for action embedding sampling of stage 2')
        parser.add_argument('--cross_stage_reg', action='store_true', 
                            help='do cross-stage score regularization if True')
        return parser
        
    def __init__(self, args, environment, actor, critic):
        super().__init__(args, environment, actor, critic)
        '''
        self.env = environment
        self.actor = actor
        self.critic = critic
        
        self.slate_size = args.slate_size
        self.noise_var = args.noise_var
        self.q_laplace_smoothness = args.q_laplace_smoothness
        self.topk_rate = args.topk_rate
        self.empty_start_rate = args.empty_start_rate
        
        self.n_item = self.env.action_space['item_id'][1]
        self.candidate_iids = np.arange(1,self.n_item+1)
        self.candidate_features = self.env.reader.get_item_list_meta(self.candidate_iids).astype(float)
        
        # replay buffer
        self.buffer = []
        self.buffer_size = args.buffer_size
        self.start_timestamp = args.start_timestamp
        '''
        self.stage2_noise_var = args.stage2_noise_var
        self.candidate_size = args.candidate_size
        self.cross_stage_reg = args.cross_stage_reg
        
    def initialize_train(self):
        '''
        Procedures before training
        '''
        action_dim_info = self.actor.get_action_dim()['separate']
        self.buffer = {
            "user_profile": np.zeros((self.buffer_size, self.env.reader.portrait_len)),
            "history_features": np.zeros((self.buffer_size, self.env.reader.max_seq_len, self.env.reader.item_vec_size)),
            "next_history_features": np.zeros((self.buffer_size, self.env.reader.max_seq_len, self.env.reader.item_vec_size)),
            "state_emb": np.zeros((self.buffer_size, self.actor.state_dim)),
            "action1_emb": np.zeros((self.buffer_size, action_dim_info[0])),
            "action2_emb": np.zeros((self.buffer_size, action_dim_info[1])),
            "action": np.zeros((self.buffer_size, self.slate_size)).astype(int),
            "reward": np.zeros((self.buffer_size,)),
            "feedback": np.zeros((self.buffer_size, self.slate_size)),
            "done": np.zeros((self.buffer_size,))
#             "critic_loss": np.zeros((self.buffer_size,))
        }
        self.buffer_head = 0
        self.current_buffer_size = 0
        self.hist_Q = [1e-6]
        self.n_stream_record = 0
        self.is_training_available = False
    
    def apply_critic(self, observation, policy_output, critic_model):
        feed_dict = {"state_emb": policy_output["state_emb"], 
                     "action1_emb": policy_output["action1_emb"], 
                     "action2_emb": policy_output["action2_emb"]}
        critic_output = critic_model(feed_dict)
        return critic_output
    
    def apply_policy(self, observation, policy_model, epsilon = 0, do_explore = False):
        feed_dict = utils.wrap_batch(observation, device = 'cpu')
        if do_explore:
            # apply action embedding noise when explore
            feed_dict['stage1_noise_var'] = self.noise_var
            feed_dict['stage2_noise_var'] = self.stage2_noise_var
            # get action embedding
            out_dict = policy_model(feed_dict)
            stage1_action_emb, stage2_action_emb = out_dict['action1_emb'], out_dict['action2_emb']
            # map to item actions
            # (B, L)
            stage1_action, stage1_action_prob = self.select_action(policy_model, stage1_action_emb, epsilon, self.candidate_size, 
                                                                   torch.FloatTensor(self.candidate_features).unsqueeze(0),
                                                                   torch.tensor(self.candidate_iids), stage = 1)
            # (B, L)
            stage2_candidate = stage1_action
            # (B, L, item_dim)
            stage2_candidate_features = self.candidate_features[stage1_action - 1]
            # (B, K)
            stage2_action, stage2_action_prob = self.select_action(policy_model, stage2_action_emb, epsilon, self.slate_size, 
                                                                   torch.FloatTensor(stage2_candidate_features),
                                                                   torch.tensor(stage2_candidate), stage = 2)
            out_dict['action'] = stage2_action
        else:
            # no exploration: no action embedding noise, topk item selection
            out_dict = policy_model(feed_dict)
        return out_dict

    def select_action(self, policy, action_emb, epsilon, K, 
                      candidate_features, candidate_ids, stage):
        '''
        Sample slate of items as actual action given action embedding
        
        @input:
        - policy: the policy model
        - action_emb: (B, action_dim)
        - epsilon: the greedy epsilon coefficient
        - K: select K items as final action
        - candidate_features: (B, L, item_dim)
        - candidate_dis: (B, L)
        - stage: the stage of cascade ranking
        '''
        # (B, L)
        action_prob = policy.score(action_emb, candidate_features, stage = stage)
        do_batch_wise_sample = (stage == 2)
        if np.random.rand() >= epsilon:
            # two types of greedy selection
            if np.random.rand() >= self.topk_rate:
                # greedy random: categorical sampling
                action = utils.sample_categorical_action(action_prob, candidate_ids, K, with_replacement = False, 
                                                         batch_wise = do_batch_wise_sample)
            else:
                # topk action
                _, indices = torch.topk(action_prob, k = K, dim = 1)
                action = torch.gather(candidate_ids, 1, indices) if do_batch_wise_sample else candidate_ids[indices]
                action = action.detach().numpy()
        else:
            # random selection for exploration
            prob = F.softmax(torch.ones_like(action_prob), dim = -1)
            action = utils.sample_categorical_action(prob, candidate_ids, K, with_replacement = False, 
                                                     batch_wise = do_batch_wise_sample)
        return action, action_prob
    
    
    def get_policy_gradient_loss(self, observation, policy_output, next_observation, actor):
        '''
        Extract supervise data from RL samples (observation, policy_output, next_observation)
        @input:
        - observation: {"user_profile": tensor, "history_features": tensor}
        - policy_output: {"state_emb": tensor, "action1_emb": tensor, "action2_emb": tensor}
        - next_observation: {"user_profile": U, "history_features": N}
        @output:
        - observation: {"user_profile": tensor , "history_features": tensor}
        - exposure: {"ids": tensor, "features": tensor}
        - user_feedback: tensor
        '''
        observation = {"user_profile": observation["user_profile"], 
                       "history_features": observation["history_features"]}
        exposed_items = policy_output["action"]
        exposure_features = torch.FloatTensor(np.array([self.env.reader.get_item_list_meta(iids.numpy()) \
                                                        for iids in exposed_items]))
        feedback = torch.FloatTensor(next_observation["previous_feedback"])
        # forward
        feed_dict = utils.wrap_batch(observation, device = 'cpu')
        out_dict = actor(feed_dict)
        stage1_action_emb, stage2_action_emb = out_dict['action1_emb'], out_dict['action2_emb']
        # map to item actions
        stage1_action_prob = actor.score(stage1_action_emb, exposure_features, stage = 1)
        stage1_behavior_loss = F.binary_cross_entropy(stage1_action_prob, feedback)
        stage2_action_prob = actor.score(stage2_action_emb, exposure_features, stage = 2)
        stage2_behavior_loss = F.binary_cross_entropy(stage2_action_prob, feedback)
        behavior_loss = stage1_behavior_loss + stage2_behavior_loss
        if self.cross_stage_reg:
            # map to item actions
            # (B, N)
            stage1_action_prob = actor.score(stage1_action_emb, 
                                             torch.FloatTensor(self.candidate_features).unsqueeze(0), 
                                             stage = 1)
            # (B, L)
            stage1_action_prob, stage1_indices = torch.topk(stage1_action_prob, k = self.candidate_size, dim = 1)
            stage2_candidates = torch.tensor(self.candidate_iids)[stage1_indices]
            # (B, L, item_dim)
            stage2_candidate_features = self.candidate_features[stage1_indices]
            # (B, L)
            stage2_action_prob = actor.score(stage2_action_emb, 
                                             torch.FloatTensor(stage2_candidate_features), 
                                             stage = 2).detach()
            # cross-stage entropy
            entropy_loss = - stage2_action_prob * torch.log(stage1_action_prob)
            behavior_loss = behavior_loss + entropy_loss
        return behavior_loss
    
    def sample_buffer(self, batch_size):
        '''
        Batch sample is organized as a tuple of (observation, policy_output, reward, next_observation)
        '''
        indices = np.random.randint(0, self.current_buffer_size, size = batch_size)
        U, H, N, S, HA1, HA2, A, R, F, D = self.read_buffer(indices)
        observation = {"user_profile": U, "history_features": H}
        policy_output = {"state_emb": S, "action1_emb": HA1, "action2_emb": HA2, "action": A}
        reward = R
        done_mask = D
        next_observation = {"user_profile": U, "history_features": N, "previous_feedback": F}
        return observation, policy_output, reward, done_mask, next_observation
    
    def update_buffer(self, observation, policy_output, reward, done_mask, next_observation, info):
        '''
        Each sample is organized as a tuple of (U,H,N,S,HA1,HA2,A,R,F,D):
        - (U)ser profile: vector
        - (H)istory: tensor
        - (N)ext history: tensor
        - (S)tate embedding: vector 
        - (H)yper-(A)ction (1) embedding: vector
        - (H)yper-(A)ction (2) embedding: vector
        - (A)ction: list of ids
        - (R)eward: scalar
        - (F)eedback: user feedback for A
        - (D)one: done flag
        '''
        for i in range(len(reward)):
            U = observation['user_profile'][i]
            H = observation['history_features'][i]
            N = self.env.reader.get_item_list_meta(info['updated_observation'][i]).astype(float)
            S = policy_output['state_emb'][i].detach().numpy() 
            HA1 = policy_output['action1_emb'][i].detach().numpy()
            HA2 = policy_output['action2_emb'][i].detach().numpy()
            A = policy_output['action'][i]
            R = reward[i]
            F = info['response'][i].detach().numpy()
            D = done_mask[i]
            self.write_buffer(self.buffer_head, (U,H,N,S,HA1,HA2,A,R,F,D))
            self.buffer_head = (self.buffer_head + 1) % self.buffer_size
            if self.n_stream_record + i < self.buffer_size:
                self.current_buffer_size += 1
        if self.current_buffer_size >= self.start_timestamp:
            self.is_training_available = True
        self.n_stream_record += len(reward)
            
    def write_buffer(self, idx, info):
        
        U, H, N, S, HA1, HA2, A, R, F, D = info
        self.buffer["user_profile"][idx] = U
        self.buffer["history_features"][idx] = H
        self.buffer["next_history_features"][idx] = N
        self.buffer["state_emb"][idx] = S
        self.buffer["action1_emb"][idx] = HA1
        self.buffer["action2_emb"][idx] = HA2
        self.buffer["action"][idx] = A
        self.buffer["reward"][idx] = R
        self.buffer["feedback"][idx] = F
        self.buffer["done"][idx] = D
        
    def read_buffer(self, indices):
        U = self.buffer["user_profile"][indices]
        H = self.buffer["history_features"][indices]
        N = self.buffer["next_history_features"][indices]
        S = self.buffer["state_emb"][indices]
        HA1 = self.buffer["action1_emb"][indices]
        HA2 = self.buffer["action2_emb"][indices]
        A = self.buffer["action"][indices]
        R = self.buffer["reward"][indices]
        F = self.buffer["feedback"][indices]
        D = self.buffer["done"][indices]
        return U, H, N, S, HA1, HA2, A, R, F, D


################
#     ML1M     #
################



