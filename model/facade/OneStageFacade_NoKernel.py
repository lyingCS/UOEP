import torch
import torch.nn.functional as F
import random
import numpy as np
import utils

from model.facade.OneStageFacade import OneStageFacade


class OneStageFacade_NoKernel(OneStageFacade):
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
        args from OneStageFacade:
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

    def apply_policy(self, observation, policy_model, epsilon=0,
                     do_explore=False, do_softmax=True):
        #         feed_dict = utils.wrap_batch(observation, device = self.device)
        feed_dict = observation
        out_dict = policy_model(feed_dict)

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

        if do_explore:
            # two types of greedy selection: uniform random or categorical random
            if np.random.rand() < epsilon:
                # greedy random: uniform sampling
                action_prob = torch.rand_like(action_prob)
            action, indices = utils.sample_categorical_action(action_prob, out_dict['candidate_ids'],
                                                              self.slate_size, with_replacement=False,
                                                              batch_wise=batch_wise, return_idx=True)
        else:
            # indices on action_prob
            _, indices = torch.topk(action_prob, k=self.slate_size, dim=1)
            # topk action
            if batch_wise:
                action = torch.gather(out_dict['candidate_ids'], 1, indices).detach()
            else:
                action = out_dict['candidate_ids'][indices].detach()
        # (B,K)
        out_dict['action'] = action
        # (B,K,item_dim)
        out_dict['action_features'] = self.candidate_features[action - 1]
        # (B,K)
        out_dict['action_prob'] = torch.gather(action_prob, 1, indices)
        # (B,L)
        out_dict['candidate_prob'] = action_prob
        return out_dict
