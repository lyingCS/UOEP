import torch
import torch.nn.functional as F
import random
import numpy as np
import utils
import itertools
import math
from model.facade.OneStageFacade import OneStageFacade


class OneStageFacade_Wolpertinger(OneStageFacade):

    def apply_policy(self, observation, policy_model, critic_model, epsilon=0,
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
        if batch_wise:
            action = torch.gather(out_dict['candidate_ids'], 1, indices).detach()  # (B, slate_size)
        else:
            action = out_dict['candidate_ids'][indices].detach()  # (B, slate_size)
        # (B,K)
        # indices, action, action_ft, action_prob = self.knn_critic_max(out_dict['state_emb'], out_dict['action_emb'],
        #                                                               out_dict['candidate_ids'],
        #                                                               out_dict['candidate_features'], policy_model,
        #                                                               critic_model)
        out_dict['action'] = action
        # (B,K,item_dim)
        out_dict['action_features'] = self.candidate_features[action - 1]
        # (B,K)
        out_dict['action_prob'] = torch.gather(action_prob, 1, indices)
        # (B,L)
        out_dict['candidate_prob'] = action_prob
        return out_dict

    def knn_critic_max(self, state_emb, action_emb, candidate_ids, candidate_features, policy, critic, k=1.5):
        with torch.no_grad():
            k = math.floor(k * self.slate_size)
            item_emb = policy.item_map(candidate_features)
            indicess, action_ids, action_fts, action_probs = [], [], [], []
            for action, state in zip(action_emb, state_emb):
                distance = -torch.sum((action - item_emb) ** 2, dim=2)
                _, indices = torch.topk(distance, k=k, dim=1)
                indices = indices[0]
                action_id = torch.gather(candidate_ids, 0, indices).detach()  # (B, slate_size)
                action_ft = candidate_features[0][action_id - 1]
                Q = []
                for i in range(k):
                    Q.append(critic(feed_dict={"state_emb": state.unsqueeze(0), "action_emb": action_ft[i].unsqueeze(0)})['q'])
                _, idx = torch.topk(torch.cat(Q), k=self.slate_size)
                final_action_prob = utils.multi_hot_encode(indices[idx], candidate_ids.shape[0]).to(self.device)
                indicess.append(indices[idx])
                action_ids.append(action_id[idx])
                action_fts.append(action_ft[idx])
                action_probs.append(final_action_prob)
            return torch.stack(indicess), torch.stack(action_ids), torch.stack(action_fts), torch.stack(action_probs)
