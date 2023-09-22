import numpy as np
import utils
import torch
import random
from argparse import Namespace

from reader.KRDataReader import KRDataReader    # wating
from model.KRUserResponse import KRUserResponse   # wating
from env.BaseRLEnvironment import BaseRLEnvironment

class KREnvironment(BaseRLEnvironment):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - urm_log_path
        from BaseEnvironment
            - env_path
            - reward_func
            - max_step_per_episode
            - initial_temper
        '''
        parser = BaseRLEnvironment.parse_model_args(parser)
#         parser = RL4RSDataReader.parse_model_args(parser)
#         parser = RL4RSUserResponse.parse_model_args(parser)
        parser.add_argument('--urm_log_path', type=str, required=True, help='log path for saved user response model')
        return parser
    
    def __init__(self, args):
        super(KREnvironment, self).__init__(args)
        infile = open(args.urm_log_path, 'r')
        class_args = eval(infile.readline())
        model_args = eval(infile.readline())
        infile.close()
        print("Loading raw data")
        self.reader = KRDataReader(model_args)
        print("Loading user response model")
        self.user_response_model = KRUserResponse(model_args, self.reader, 'cpu')
        checkpoint = torch.load(model_args.model_path + ".checkpoint")
        self.user_response_model.load_state_dict(checkpoint["model_state_dict"])
        
        # spaces
        stats = self.reader.get_statistics()
        self.action_space = {'item_id': ('nominal', stats['n_item']), 
                             'item_feature': ('continuous', stats['item_vec_size'], 'normal')}
        self.observation_space = {'user_profile': ('continuous', stats['user_portrait_len'], 'positive'), 
                                  'history': ('sequence', stats['max_seq_len'], ('continuous', stats['item_vec_size']))}
        
    def reset(self, params = {'batch_size': 1, "empty_history": True}):
        '''
        @output:
        - observation: {'user_profile': (B, portrait_dim), 
                        'history': (B, H), 
                        'history_feature': (B, H, item_dim)}
        '''
        if 'empty_history' not in params:
            params['empty_history'] = True
        BS = params['batch_size']
        observation = super().reset(params)
        if 'pick_rows' in params:
            sample_info = self.pick_user(params['pick_rows'], empty_history = params['empty_history'])
        else:
            sample_info = self.sample_user(BS, empty_history = params['empty_history'])
        self.interaction_records = {
            'user_profile': sample_info['user_profile'],
            'history': sample_info['history'],
            'action_history': [[] for _ in range(BS)],
            'action_feedback': [[] for _ in range(BS)],
            'reward': [[0] for _ in range(BS)]
        }
        self.continue_flag = [True] * BS
        self.temper = np.array([self.initial_temper] * BS).astype(float)
        observation.update({'user_profile': sample_info['user_profile'], 
                            'history_features': sample_info['history_features'], 
                            'reward': [utils.padding_and_clip(self.interaction_records['reward'][i], self.max_step_per_episode) 
                                 for i in range(BS)]})
        return observation
    
    def sample_user(self, n_user, empty_history = False):
        '''
        sample random users and their histories
        '''
        random_rows = np.random.randint(0, len(self.reader.data['train']), n_user)
        return self.pick_user(random_rows, empty_history)

    def pick_user(self, rows, empty_history = False):
        raw_portrait = [self.reader.user_meta[self.reader.data['train']['user_id'][rowid]] for rowid in rows]
        # raw_portrait = [eval(self.reader.data['train']['user_protrait'][rowid]) for rowid in random_rows]
        # portrait = np.log(np.array(raw_portrait)+1)
        # print("type(portrait): ", type(portrait))
        # print("type(raw_portrait): ", type(raw_portrait))
        
        portrait = np.array(raw_portrait)
        # print(portrait)
        history = []
        history_features = []
        for rowid in rows:
            H = [] if empty_history else eval(f"{self.reader.data['train']['user_mid_history'][rowid]}")
            H = utils.padding_and_clip(H, self.reader.max_seq_len)
            history.append(H)
            # print("H: ", H)
            history_features.append(self.reader.get_item_list_meta(H).astype(float))
        return {'user_profile': portrait, 
                'history': history, 
                'history_features': np.array(history_features)}
    
    def step(self, step_dict):
        '''
        @input:
        - step_dict: {'action': list of list of item id, size (B, slate_size)}
        '''
        # actions (exposures)
        action = step_dict['action'] # (B, slate_size), should be item ids only
        action_features = np.array([self.reader.get_item_list_meta(slate) for slate in action])
        # wrap input for user response model (URM)
        current_observation = self.get_active_observation()
        batch_data = {
            'user_profile': current_observation['user_profile'],
            'history_features': current_observation['history_features'],
            'exposure_features': action_features
        }
        # URM forward
        output_dict = self.user_response_model(utils.wrap_batch(batch_data, device = 'cpu'))
        # print("output_dict: ", output_dict)
        response = torch.bernoulli(output_dict['probs']) # (B, slate_size)
#         pos_ratio = (torch.sum(response, dim = 1) + 1) / (response.shape[1] + 1)
        temper_down = torch.sum(response, dim = 1) - response.shape[1] - 1
        new_clicks = [[action[i,j] \
                           for j,resp in enumerate(user_resp) if resp == 1] \
                              for i,user_resp in enumerate(response)]
        # reward (B,)
        reward = self.reward_func(response).detach().numpy()
        # update user history records, temper, and continue flags
        idx = 0
        updated_observation = []
        done_mask = [False] * len(reward)
        for i,flag in enumerate(self.continue_flag):
            if flag: # activate user
#                 self.temper[i] = pos_ratio[idx] * self.temper[i] - 1 # reduce temper according to number user actions
                self.temper[i] = self.temper[i] + temper_down[idx]
                self.interaction_records['history'][i] += new_clicks[idx]
                self.interaction_records['reward'][i].append(reward[idx])
                updated_observation.append(self.interaction_records['history'][i][-self.reader.max_seq_len:])
                if self.temper[i] < 1:
                    self.continue_flag[i] = False
                    done_mask[idx] = True
                idx += 1
        # new_observation
        new_observation = self.get_active_observation()
        # done
#         done = True if len(new_observation['user_profile']) == 0 else False
        done = done_mask
        
        return new_observation, reward, done, {'updated_observation': np.array(updated_observation), 
                                               'response': response}
    
    def get_active_observation(self):
        '''
        Get observation records for activate users
        
        Note: use self.continue_flag to determine active users
        '''
        # user profiles
        profiles = self.interaction_records['user_profile'][self.continue_flag]
        # user history
        user_hist = [self.reader.get_item_list_meta(H[-self.reader.max_seq_len:]).astype(float) \
                         for i,H in enumerate(self.interaction_records['history']) \
                             if self.continue_flag[i]]
        user_hist_feature = np.array(user_hist)
        # user rewards
        reward_history = [utils.padding_and_clip(R, self.max_step_per_episode) 
                            for i,R in enumerate(self.interaction_records['reward']) \
                                if self.continue_flag[i]]
        observation = {'user_profile': profiles, 
                       'history_features': user_hist_feature, 
                       'reward': np.array(reward_history)}
        return observation