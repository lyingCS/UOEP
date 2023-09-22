import torch
import torch.nn as nn

from model.policy.OneStagePolicy import OneStagePolicy
from model.components import DNN
from model.score_func import *

class OneStagePolicy_with_WideDeep(OneStagePolicy):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - from OneStagePolicy:
            - state_encoder_feature_dim
            - state_encoder_attn_n_head
            - state_encoder_hidden_dims
            - state_encoder_dropout_rate
        - policy_action_hidden
        - policy_scorer_hidden
        '''
        parser = OneStagePolicy.parse_model_args(parser)
        parser.add_argument('--policy_action_hidden', type=int, nargs='+', default=[128], 
                            help='hidden dim of the action net')
        parser.add_argument('--policy_scorer_hidden', type=int, default=64, 
                            help='hidden dim of the action net')
        return parser
    
    def __init__(self, args, environment):
        '''
        action_space = {'item_id': ('nominal', stats['n_item']), 
                        'item_feature': ('continuous', stats['item_vec_size'], 'normal')}
        observation_space = {'user_profile': ('continuous', stats['user_portrait_len'], 'positive']), 
                            'history': ('sequence', stats['max_seq_len'], ('continuous', stats['item_vec_size']))}
        '''
        super().__init__(args, environment)
        self.policy_scorer_hidden = args.policy_scorer_hidden
        # action is the set of parameters of wide&deep scorer
        self.hidden_dim = args.policy_scorer_hidden
        self.action_dim = (self.item_dim + 2) * (self.hidden_dim + 1)
#         self.action_layer = nn.Linear(self.state_dim, self.action_dim)
        self.action_layer = DNN(self.state_dim, args.policy_action_hidden, self.action_dim, 
                                dropout_rate = self.dropout_rate, do_batch_norm = True)
        self.scorer_norm = nn.LayerNorm(args.policy_scorer_hidden)
        
    def score(self, action_emb, item_emb, do_softmax = True):
        output = wide_and_deep_scorer(action_emb, item_emb, self.item_dim, self.policy_scorer_hidden, 
                                      self.scorer_norm, self.dropout_rate)
        if do_softmax:
            return torch.softmax(output, dim = -1)
        else:
            return output