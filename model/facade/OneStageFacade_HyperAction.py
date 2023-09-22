import torch
import torch.nn.functional as F
import random
import numpy as np
import utils

from model.facade.OneStageFacade import OneStageFacade

class OneStageFacade_HyperAction(OneStageFacade):
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
        parser = OneStageFacade.parse_model_args(parser)
        return parser
        
    def __init__(self, args, environment, actor, critic):
        super().__init__(args, environment, actor, critic)
    
    def apply_policy(self, observation, policy_model, epsilon = 0, 
                     do_explore = False, do_softmax = True):
#         feed_dict = utils.wrap_batch(observation, device = self.device)
        feed_dict = observation
        out_dict = policy_model(feed_dict)
        if do_explore:
            action_emb = out_dict['action_emb']
            # sampling noise of action embedding
            if np.random.rand() < epsilon:
                action_emb = torch.clamp(torch.rand_like(action_emb)*self.noise_var, -1, 1)
            else:
                action_emb = action_emb + torch.clamp(torch.rand_like(action_emb)*self.noise_var, -1, 1)
#                 self.noise_var -= self.noise_decay
            out_dict['action_emb'] = action_emb
        out_dict['Z'] = out_dict['action_emb']
            
        if 'candidate_ids' in feed_dict:
            out_dict['candidate_features'] = feed_dict['candidate_features']
            out_dict['candidate_ids'] = feed_dict['candidate_ids']
            batch_wise = True
        else:
            out_dict['candidate_features'] = self.candidate_features.unsqueeze(0)
            out_dict['candidate_ids'] = self.candidate_iids
            batch_wise = False
            
        # action prob (B,L)
        action_prob = policy_model.score(out_dict['action_emb'], 
                                         out_dict['candidate_features'], 
                                         do_softmax = do_softmax)

        # two types of greedy selection
        if do_explore and np.random.rand() >= self.topk_rate:
            # greedy random: categorical sampling
            action, indices = utils.sample_categorical_action(action_prob, out_dict['candidate_ids'], 
                                                              self.slate_size, with_replacement = False, 
                                                              batch_wise = batch_wise, return_idx = True)
        else:
            _, indices = torch.topk(action_prob, k = self.slate_size, dim = 1)
            # topk action
            if batch_wise:
                action = torch.gather(out_dict['candidate_ids'], 1, indices).detach() # (B, slate_size)
            else:
                action = out_dict['candidate_ids'][indices].detach() # (B, slate_size)
        out_dict['action'] = action 
        out_dict['action_features'] = self.candidate_features[indices]
        out_dict['action_prob'] = torch.gather(action_prob, 1, indices) 
        out_dict['candidate_prob'] = action_prob
        return out_dict
    
    def infer_hyper_action(self, observation, policy_output, actor):
        '''
        inverse function or pooling for A --> Z
        '''
        # (B,K)
        A = policy_output['action'] 
        # (B,K,item_dim)
        item_embs = self.candidate_features[A-1]
        # (B,K,kernel_dim)
        Z = torch.mean(actor.item_map(item_embs).view(A.shape[0],A.shape[1],-1), dim = 1)
        return {'Z': Z, 'action_emb': Z, 'state_emb': policy_output['state_emb']}

    def apply_critic(self, observation, policy_output, critic_model):
        feed_dict = {"state_emb": policy_output["state_emb"], 
                     "action_emb": policy_output["action_emb"]}
        critic_output = critic_model(feed_dict)
        return critic_output