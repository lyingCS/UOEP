import torch
import torch.nn.functional as F
import random
import numpy as np
import utils

from model.components import DNN
from model.facade.OneStageFacade import OneStageFacade

class OneStageFacade_LatentAction(OneStageFacade):
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
        parser.add_argument('--inverse_hidden_dims', type=int, nargs='+', default=[128], 
                            help='hidden dims of the inverse module')
        parser.add_argument('--inverse_dropout_rate', type=float, default=0.1, 
                            help='dropout rate of the inverse module')
        return parser
        
    def __init__(self, args, environment, actor, critic):
        super().__init__(args, environment, actor, critic)
        self.inverse_model = DNN(actor.state_dim * 2, args.inverse_hidden_dims, actor.action_dim, 
                                 dropout_rate = args.inverse_dropout_rate, do_batch_norm = True).to(args.device)
        
    
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
    
    def infer_latent(self, current_observation, next_observation, actor):
        '''
        inverse function or pooling for A --> Z
        '''
        # (B,state_dim)
        current_state = self.apply_policy(current_observation, actor)['state_emb'].detach()
        # (B,state_dim)
        next_state = self.apply_policy(next_observation, actor)['state_emb'].detach()
        # (B,latent_dim)
        Z = self.inverse_model(torch.cat((current_state, next_state), dim = 1))
        # (1,L,item_dim)
        candidate_embs = self.candidate_features.unsqueeze(0)
        # (B,L)
        candidate_score = actor.score(Z, candidate_embs, do_softmax = True)
        return {'Z': Z, 'action_emb': Z, 'candidate_score': candidate_score}

    def apply_critic(self, observation, policy_output, critic_model):
        feed_dict = {"state_emb": policy_output["state_emb"], 
                     "action_emb": policy_output["action_emb"]}
        critic_output = critic_model(feed_dict)
        return critic_output