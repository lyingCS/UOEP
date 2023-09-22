import numpy as np
import utils
import torch
import random
from copy import deepcopy
from argparse import Namespace
from torch.utils.data import DataLoader

from reader.ML1MDataReader import ML1MDataReader    # wating
from model.ML1MUserResponse import ML1MUserResponse   # wating
from env.BaseRLEnvironment import BaseRLEnvironment


class ML1MEnvironment_GPU(BaseRLEnvironment):
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
        parser.add_argument('--temper_sweet_point', type=float, default=0.9, help='between [0,1.0]')
        parser.add_argument('--temper_prob_lag', type=float, default=100, help='smaller value means larger probability change from temper (min = 0)')
        return parser
    
    def __init__(self, args):
        super(ML1MEnvironment_GPU, self).__init__(args)
        self.temper_sweet_point = args.temper_sweet_point
        self.temper_prob_lag = args.temper_prob_lag
        
        infile = open(args.urm_log_path, 'r')
        class_args = eval(infile.readline()) # example: Namespace(model='RL4RSUserResponse', reader='RL4RSDataReader')
        model_args = eval(infile.readline()) # model parameters in Namespace
        print("Environment arguments: \n" + str(model_args))
        infile.close()
        print("Loading raw data")
        assert class_args.reader == 'ML1MDataReader' and class_args.model == 'ML1MUserResponse'
        self.reader = ML1MDataReader(model_args)
        print("Loading user response model")
        self.user_response_model = ML1MUserResponse(model_args, self.reader, args.device)
        self.user_response_model.load_from_checkpoint(model_args.model_path, with_optimizer = False)
        self.user_response_model.to(args.device)
        
        # spaces
        stats = self.reader.get_statistics()
        self.action_space = {'item_id': ('nominal', stats['n_item']), 
                             'item_feature': ('continuous', stats['item_vec_size'], 'normal')}
        self.observation_space = {'user_profile': ('continuous', stats['user_portrait_len'], 'positive'), 
                                  'history': ('sequence', stats['max_seq_len'], ('continuous', stats['item_vec_size']))}
        
        # buffered user samples
#         self.user_sample_buffer_size = 1024
#         self.user_samples = self.refresh_user_samples(self.user_sample_buffer_size)
        
#     def refresh_user_samples(self, B = 1024):
#         '''
#         batch from reader:
#         {
#             'timestamp': (B,), 
#             'exposure': (B,K) 
#             'exposure_features': (B,K,item_dim)
#             'feedback': (B,K)
#             'history': (B,H)
#             'history_features': (B,H,item_dim) 
#             'history_features': (B,) 
#             'user_profile': (B,user_dim) 
#         }
#         '''
#         self.reader.set_phase("train")
#         train_loader = DataLoader(self.reader, batch_size = B, shuffle = True, pin_memory = True, num_workers = 8)
#         for i,batch_data in enumerate(train_loader):
#             wrapped_batch = utils.wrap_batch(batch_data, device = self.user_response_model.device)
#             break
#         self.user_sample_head = 0
#         return wrapped_batch
        
    
#     def sample_user(self, n_user, empty_history = False, pick_rows = []):
#         '''
#         sample random users and their histories
#         @output:
#         - batch: {"user_profile": np array (B, user_dim), 
#                     "history": (B, H),
#                     "history_features": (B, H, item_dim),
#                     "exposed_items": (B, K),
#                     "exposed_item_features": (B, K, item_dim),
#                     "feedback": (B, K)}
#         '''
#         if len(pick_rows) == 0:
# #             pick_rows = torch.arange(self.user_sample_head, self.user_sample_head + n_user).to(self.user_response_model.device)
#             if self.user_sample_head + n_user >= self.user_sample_buffer_size:
#                 self.refresh_user_samples(self.user_sample_buffer_size)
#             batch = {k:v[self.user_sample_head:self.user_sample_head + n_user] for k,v in self.user_samples.items()}
#             if empty_history:
#                 batch['history'] *= 0
#                 batch['history_features'] *= 0
#             self.user_sample_head += n_user
#         else:
#             raise NotImplemented
#         return batch
        
    def reset(self, params = {'batch_size': 1, 'empty_history': True}):
        '''
        Reset environment with new sampled users
        @input:
        - params: {'batch_size': scalar, 
                    'empty_history': True if start from empty history, 
                    'initial_history': start with initial history, mu }
        @output:
        - observation: {'user_profile': (B, portrait_dim), 
                        'history': (B, H), 
                        'history_feature': (B, H, item_dim)
                        'history_feedback': (B, H, item_dim)}
        '''
        self.empty_history_flag = params['empty_history'] if 'empty_history' not in params else True
        BS = params['batch_size']
        observation = {'batch_size': BS}
        if 'sample' in params:
            sample_info = params['sample']
        elif 'pick_rows' in params:
            sample_info = self.reader.get_row_data(params['pick_rows'])
            sample_info = utils.wrap_batch(sample_info, device=self.user_response_model.device)
        else:
            self.batch_iter = iter(DataLoader(self.reader, batch_size = BS, shuffle = True, 
                                              pin_memory = True, num_workers = 2))
            sample_info = next(self.batch_iter)
            sample_info = utils.wrap_batch(sample_info, device = self.user_response_model.device)
        self.current_observation = {
            'user_profile': sample_info['user_profile'],  # (B, user_dim)
            'history': sample_info['history'],  # (B, H)
            'history_features': sample_info['history_features'], # (B, H, item_dim)
            'cummulative_reward': torch.zeros(BS).to(self.user_response_model.device),
            'temper': torch.ones(BS).to(self.user_response_model.device) * self.initial_temper,
            'step': torch.zeros(BS).to(self.user_response_model.device),
        }
        self.reward_history = [0.]
        self.step_history = [0.]
        return deepcopy(self.current_observation)
        
    
    def step(self, step_dict):
        '''
        @input:
        - step_dict: {'action': (B, slate_size),
                        'action_features': (B, slate_size, item_dim) }
        '''
        # actions (exposures)
        action = step_dict['action'] # (B, slate_size), should be item ids only
        action_features = step_dict['action_features']
        batch_data = {
            'user_profile': self.current_observation['user_profile'],
            'history_features': self.current_observation['history_features'],
            'exposure_features': action_features
        }
        # URM forward
        with torch.no_grad():
            output_dict = self.user_response_model(batch_data)
#             output_dict = self.user_response_model(utils.wrap_batch(batch_data, device = self.user_response_model.device))
            response = torch.bernoulli(output_dict['probs']) # (B, slate_size)
#             prob_scale = (self.current_observation['temper'].clone().detach().view(-1,1) + self.temper_prob_lag) / (self.initial_temper + self.temper_prob_lag)
            probs_under_temper = output_dict['probs'] # * prob_scale
            response = torch.bernoulli(probs_under_temper).detach() # (B, slate_size)

            # reward (B,)
            immediate_reward = self.reward_func(response).detach()

            # (B, H+slate_size)
            H_prime = torch.cat((self.current_observation['history'], action), dim = 1) 
            # (B, H+slate_size, item_dim)
            H_prime_features = torch.cat((self.current_observation['history_features'], action_features), dim = 1) 
            # (B, H+slate_size)
            F_prime = torch.cat((torch.ones_like(self.current_observation['history']), response), dim = 1).to(torch.long) 
            # vector, vector
            row_indices, col_indices = (F_prime == 1).nonzero(as_tuple=True) 
            # (B,), the number of positive iteraction as history length
            L = F_prime.sum(dim = 1) 
            
            # user history update
            offset = 0
            newH = torch.zeros_like(self.current_observation['history'])
            newH_features = torch.zeros_like(self.current_observation['history_features'])
            for row_id in range(action.shape[0]):
                right = offset + L[row_id]
                left = right - self.reader.max_seq_len
                newH[row_id] = H_prime[row_id, col_indices[left:right]]
                newH_features[row_id] = H_prime_features[row_id,col_indices[left:right],:]
                offset += L[row_id]
            self.current_observation['history'] = newH
            self.current_observation['history_features'] = newH_features
            self.current_observation['cummulative_reward'] += immediate_reward

            # temper update for leave model
            temper_down = (-immediate_reward+1) * response.shape[1] + 1
#             temper_down = -(torch.sum(response, dim = 1) - response.shape[1] - 1)
#             temper_down = torch.abs(torch.sum(response, dim = 1) - response.shape[1] * self.temper_sweet_point) + 1
            self.current_observation['temper'] -= temper_down
            # leave signal
            done_mask = self.current_observation['temper'] < 1
            # step update
            self.current_observation['step'] += 1

            # update rows where user left
#             refresh_rows = done_mask.nonzero().view(-1)
#             print(f"#refresh: {refresh_rows}")
            if done_mask.sum() > 0:
                final_rewards = self.current_observation['cummulative_reward'][done_mask].detach().cpu().numpy()
                final_steps = self.current_observation['step'][done_mask].detach().cpu().numpy()
                self.reward_history.append(final_rewards[-1])
                self.step_history.append(final_steps[-1])
                # sample new users to fill in the blank
                new_sample_flag = False
                try:
                    sample_info = next(self.iter)
                    if sample_info['user_profile'].shape[0] != done_mask.shape[0]:
                        new_sample_flag = True
                except:
                    new_sample_flag = True
                if new_sample_flag:
                    self.iter = iter(DataLoader(self.reader, batch_size = done_mask.shape[0], shuffle = True, 
                                                pin_memory = True, num_workers = 2))
                    sample_info = next(self.iter)
                sample_info = utils.wrap_batch(sample_info, device = self.user_response_model.device)
                for obs_key in ['user_profile', 'history', 'history_features']:
                    self.current_observation[obs_key][done_mask] = sample_info[obs_key][done_mask]
                self.current_observation['cummulative_reward'][done_mask] *= 0
                self.current_observation['temper'][done_mask] *= 0
                self.current_observation['temper'][done_mask] += self.initial_temper
            self.current_observation['step'][done_mask] *= 0
#         print(f"step: {self.current_observation['step']}")
        return deepcopy(self.current_observation), immediate_reward, done_mask, {'response': response}

    def step_without_new_sample(self, step_dict):
        '''
        @input:
        - step_dict: {'action': (B, slate_size),
                        'action_features': (B, slate_size, item_dim) }
        '''
        # actions (exposures)
        action = step_dict['action']  # (B, slate_size), should be item ids only
        action_features = step_dict['action_features']
        batch_data = {
            'user_profile': self.current_observation['user_profile'],
            'history_features': self.current_observation['history_features'],
            'exposure_features': action_features
        }
        # URM forward
        with torch.no_grad():
            output_dict = self.user_response_model(batch_data)
            #             output_dict = self.user_response_model(utils.wrap_batch(batch_data, device = self.user_response_model.device))
            response = torch.bernoulli(output_dict['probs'])  # (B, slate_size)
            #             prob_scale = (self.current_observation['temper'].clone().detach().view(-1,1) + self.temper_prob_lag) / (self.initial_temper + self.temper_prob_lag)
            probs_under_temper = output_dict['probs']  # * prob_scale
            response = torch.bernoulli(probs_under_temper).detach()  # (B, slate_size)

            # reward (B,)
            immediate_reward = self.reward_func(response).detach()

            # (B, H+slate_size)
            H_prime = torch.cat((self.current_observation['history'], action), dim=1)
            # (B, H+slate_size, item_dim)
            H_prime_features = torch.cat((self.current_observation['history_features'], action_features), dim=1)
            # (B, H+slate_size)
            F_prime = torch.cat((torch.ones_like(self.current_observation['history']), response), dim=1).to(
                torch.long)
            # vector, vector
            row_indices, col_indices = (F_prime == 1).nonzero(as_tuple=True)
            # (B,), the number of positive iteraction as history length
            L = F_prime.sum(dim=1)

            # user history update
            offset = 0
            newH = torch.zeros_like(self.current_observation['history'])
            newH_features = torch.zeros_like(self.current_observation['history_features'])
            for row_id in range(action.shape[0]):
                right = offset + L[row_id]
                left = right - self.reader.max_seq_len
                newH[row_id] = H_prime[row_id, col_indices[left:right]]
                newH_features[row_id] = H_prime_features[row_id, col_indices[left:right], :]
                offset += L[row_id]
            self.current_observation['history'] = newH
            self.current_observation['history_features'] = newH_features
            done_mask = self.current_observation['temper'] < 1
            # step update
            self.current_observation['step'] += 1-done_mask.int()
            self.current_observation['cummulative_reward'] += immediate_reward * (1-done_mask.int())

            # temper update for leave model
            temper_down = (-immediate_reward + 1) * response.shape[1] + 1
            #             temper_down = -(torch.sum(response, dim = 1) - response.shape[1] - 1)
            #             temper_down = torch.abs(torch.sum(response, dim = 1) - response.shape[1] * self.temper_sweet_point) + 1
            self.current_observation['temper'] -= temper_down
            # leave signal
            done_mask = self.current_observation['temper'] < 1

            # update rows where user left
            #             refresh_rows = done_mask.nonzero().view(-1)
            #             print(f"#refresh: {refresh_rows}")
            if done_mask.sum() == done_mask.shape[0]:
                final_rewards = self.current_observation['cummulative_reward'][done_mask].detach().cpu().numpy()
                final_steps = self.current_observation['step'][done_mask].detach().cpu().numpy()
                self.reward_history.extend(list(final_rewards))
                self.step_history.extend(list(final_steps))
        #         print(f"step: {self.current_observation['step']}")
        return deepcopy(self.current_observation), immediate_reward, done_mask, {'response': response}, [
            self.reward_history[1:], self.step_history[1:]]

    def stop(self):
        self.iter = None
    
    def get_new_iterator(self, B):
        return iter(DataLoader(self.reader, batch_size = B, shuffle = True, 
                               pin_memory = True, num_workers = 2))
    
