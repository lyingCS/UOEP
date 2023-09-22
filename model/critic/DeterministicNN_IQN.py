import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import math

from model.components import DNN
from utils import get_regularization


class DeterministicNN_IQN(nn.Module):

    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - critic_hidden_dims
        - critic_dropout_rate
        '''
        parser.add_argument('--critic_hidden_dims', type=int, nargs='+', default=[128],
                            help='specificy a list of k for top-k performance')
        parser.add_argument('--critic_dropout_rate', type=float, default=0.2,
                            help='dropout rate in deep layers')
        parser.add_argument('--critic_hidden_f_dims', type=int, nargs='+', default=[32],
                            help='hidden_dims')
        parser.add_argument('--embedding_dim', type=int, default=16,
                            help='embedding dim')
        parser.add_argument('--tau_embed_dim', type=int, default=1,
                            help='embed_dim of tau')
        return parser

    def __init__(self, args, environment, policy):
        super().__init__()
        self.state_dim = policy.state_dim
        self.action_dim = policy.action_dim
        self.tau_embed_dim = args.tau_embed_dim
        #         self.state_encoder = policy.state_encoder
        self.state_action_net = nn.Sequential(DNN(self.state_dim + self.action_dim, args.critic_hidden_dims, args.embedding_dim,
                                dropout_rate=args.critic_dropout_rate, do_batch_norm=True), nn.ReLU())
        if self.tau_embed_dim > 1:
            self.i_ = torch.Tensor(np.arange(args.tau_embed_dim))
        self.head_tau = nn.Sequential(DNN(args.tau_embed_dim, [], args.embedding_dim,
                                                  dropout_rate=args.critic_dropout_rate, do_batch_norm=True), nn.ReLU())
        self.output_layer = DNN(args.embedding_dim, args.critic_hidden_f_dims, 1,
                                dropout_rate=args.critic_dropout_rate, do_batch_norm=True)

    def forward(self, feed_dict, tau_quantile):
        '''
        @input:
        - feed_dict: {'state_emb': (B, state_dim), 'action_emb': (B, action_dim)}
        '''
        state_emb = feed_dict['state_emb']
        #         state_emb = self.state_encoder(feed_dict)['state_emb'].view(-1, self.state_dim)
        action_emb = feed_dict['action_emb'].view(-1, self.action_dim)
        Q = self.state_action_net(torch.cat((state_emb, action_emb), dim=-1))
        #         reg = get_regularization(self.net, self.state_encoder)
        if self.tau_embed_dim > 1:
            a = torch.cos(torch.Tensor([math.pi])*self.i_*tau_quantile)
        else:
            a = tau_quantile
        tau_output = self.head_tau(a)  # [batch_size x embedding_dim]

        output = self.output_layer(
            torch.mul(Q, tau_output)
        ).view(-1, 1)
        reg = get_regularization(self.state_action_net) + get_regularization(self.head_tau) + \
              get_regularization(self.output_layer)
        return {'q': output, 'reg': reg}

    def get_sampled_Z(self, observation, confidences, policy_output):
        """Runs IQN for K different confidence levels
        Parameters
        ----------
        # state: torch.Tensor [batch_size x dim_state]
        confidences: torch.Tensor. [1 x K]
        Returns
        -------
        Z_tau_K: torch.Tensor [batch_size x K]

        """
        K = confidences.size(0)  # number of confidence levels to evaluate
        batch_size = policy_output['state_emb'].size(0) if policy_output['state_emb'].dim() > 1 else 1
        # Reorganize so that the NN runs per one quantile at a time. Repeat
        # all batch_size block "num_quantiles" times:
        # [batch_size * K, dim_state]
        # x = state.repeat(1, K).view(-1, self.dim_state)
        # [batch_size * K, dim_state]
        # a = action.repeat(1, K).view(-1, self.dim_action)
        x = policy_output['state_emb'].repeat(1, K).view(-1, self.state_dim)
        a = policy_output['action_emb'].repeat(1, K).view(-1, self.action_dim)
        y = confidences.repeat(batch_size, 1).view(
            K*batch_size, 1)  # [batch_size * K, 1]
        Z_tau_K = self(feed_dict={'state_emb': x, 'action_emb': a}, tau_quantile=y)['q'].view(batch_size, K)
        return Z_tau_K


