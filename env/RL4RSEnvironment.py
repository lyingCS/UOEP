import numpy as np
import utils
import torch
import random
from argparse import Namespace

from reader.RL4RSDataReader import RL4RSDataReader
from model.RL4RSUserResponse import RL4RSUserResponse
from env.BaseRLEnvironment import BaseRLEnvironment

class RL4RSEnvironment(BaseRLEnvironment):
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
        parser.add_argument('--urm_log_path', type=str, required=True, help='log path for saved user response model')
        return parser
    
    def __init__(self, args):
        super(RL4RSEnvironment, self).__init__(args)
        infile = open(args.urm_log_path, 'r')
        class_args = eval(infile.readline()) # example: Namespace(model='RL4RSUserResponse', reader='RL4RSDataReader')
        model_args = eval(infile.readline()) # model parameters in Namespace
        infile.close()
        print("Loading raw data")
        assert class_args.reader == 'RL4RSDataReader' and class_args.model == 'RL4RSUserResponse'
        self.reader = RL4RSDataReader(model_args)
        print("Loading user response model")
        self.user_response_model = RL4RSUserResponse(model_args, self.reader, 'cpu')
        checkpoint = torch.load(model_args.model_path + '.checkpoint')
        self.user_response_model.load_state_dict(checkpoint['model_state_dict'])
        
        # spaces
        stats = self.reader.get_statistics()
        self.action_space = {'item_id': ('nominal', stats['n_item']), 
                             'item_feature': ('continuous', stats['item_vec_size'], 'normal')}
        self.observation_space = {'user_profile': ('continuous', stats['user_portrait_len'], 'positive'), 
                                  'history': ('sequence', stats['max_seq_len'], ('continuous', stats['item_vec_size']))}
        
    def reset(self, params = {'batch_size': 1, 'empty_history': True, 'pick_rows': []}):
        '''
        Reset environment with new sampled users
        @input:
        - params: {'batch_size': scalar, 'empty_history': True if start from empty history, 
                    'initial_history': start with initial history, mu }
        @output:
        - observation: {'user_profile': (B, portrait_dim), 
                        'history': (B, H), 
                        'history_feature': (B, H, item_dim)}
        '''
        BS = params['batch_size']
        observation = super().reset(params)
        if 'initial_history' in params:
            sample_info = params['initial_history']
        else:
            if 'empty_history' not in params:
                params['empty_history'] = True
            sample_info = self.sample_user(BS, empty_history = params['empty_history'], pick_rows = params['pick_rows'])
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
    
    def sample_user(self, n_user, empty_history = False, with_feedback = False, pick_rows = []):
        '''
        sample random users and their histories
        '''
        if len(pick_rows) > 0:
            random_rows = pick_rows
        else:
            random_rows = np.random.randint(0, len(self.reader.data['train']), n_user)
        raw_portrait = [eval(self.reader.data['train']['user_protrait'][rowid]) for rowid in random_rows]
        portrait = np.log(np.array(raw_portrait)+1)
        history = []
        history_features = []
        for rowid in random_rows:
            H = [] if empty_history else eval(f"[{self.reader.data['train']['user_seqfeature'][rowid]}]")
            H = utils.padding_and_clip(H, self.reader.max_seq_len)
            history.append(H)
            history_features.append(self.reader.get_item_list_meta(H).astype(float))
        batch = {'user_profile': portrait, 
                 'history': history, 
                 'history_features': np.array(history_features)}
        if with_feedback:
            exposed_items = []
            exposed_item_features = []
            feedback = []
            for rowid in random_rows:
                ex = eval(self.reader.data['train']['exposed_items'][rowid])
                exposed_items.append(ex)
                exposed_item_features.append(self.reader.get_item_list_meta(ex))
                feedback.append(eval(self.reader.data['train']['user_feedback'][rowid]))
            batch['exposed_items'] = exposed_items
            batch['exposed_item_features'] = np.array(exposed_item_features)
            batch['feedback'] = feedback
        return batch
    
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
#         probs_after_temper = output_dict['probs'] * torch.tensor(self.temper[self.continue_flag]).view(-1,1) / self.initial_temper
        response = torch.bernoulli(output_dict['probs']) # (B, slate_size)
#         response = torch.bernoulli(probs_after_temper) # (B, slate_size)
        # user clicks as response
        new_clicks = [[action[i,j] \
                           for j,resp in enumerate(user_resp) if resp == 1] \
                              for i,user_resp in enumerate(response)]
        # reward (B,)
        reward = self.reward_func(response).detach().numpy()
        # update user history records, temper, and continue flags
        temper_down = torch.sum(response, dim = 1) - response.shape[1] - 1
        idx = 0 # active user index
        updated_observation = [] # updated observation for both done and not done users
        done_mask = [False] * len(reward)
        for i,flag in enumerate(self.continue_flag):
            if flag: # activate user
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
        done = done_mask
        
        return new_observation, reward, done, {'updated_observation': np.array(updated_observation), 
                                               'response': response}
    
    def get_active_observation(self):
        '''
        Get observation records for activate (not done) users
        
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
    