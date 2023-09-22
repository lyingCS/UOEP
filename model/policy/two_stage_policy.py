import torch.nn.functional as F
import torch.nn as nn
import torch
import math

from model.components import DNN
from model.policy.OneStagePolicy import BaseStateEncoder
from utils import get_regularization
from model.score_func import *
    
class TwoStagePolicy(nn.Module):
    '''
    Stage one scorer: linear
    Stage two scorer: 3-layer DNN with residual link
    '''
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - from BaseStateEncoder:
            - state_encoder_feature_dim
            - state_encoder_attn_n_head
            - state_encoder_hidden_dims
            - state_encoder_dropout_rate
        '''
        parser = BaseStateEncoder.parse_model_args(parser)
#         parser.add_argument('--stage2_scorer_hidden_dims', type=int, nargs='+', default=[64,32], 
#                             help='hidden layer dimensions in scorer deep layers')
        parser.add_argument('--stage2_scorer_hidden_dim', type=int, default=256, 
                            help='hidden layer dimension in scorer deep layers')
#         parser.add_argument('--stage2_dropout_rate', type=float, default=0.2, 
#                             help='dropout rate in scorer deep layers')
        return parser
    
    def __init__(self, args, environment):
        '''
        action_space = {'item_id': ('nominal', stats['n_item']), 
                        'item_feature': ('continuous', stats['item_vec_size'], 'normal')}
        observation_space = {'user_profile': ('continuous', stats['user_portrait_len'], 'positive']), 
                            'history': ('sequence', stats['max_seq_len'], ('continuous', stats['item_vec_size']))}
        '''
        super().__init__()
        self.dropout_rate = args.state_encoder_dropout_rate
        # action space
        self.item_space = environment.action_space['item_id'][1]
        self.item_dim = environment.action_space['item_feature'][1]
        # policy network modules
        self.state_encoder = BaseStateEncoder(args, environment)
        self.state_dim = self.state_encoder.state_dim
        # stage 1 action is the linear scorer parameters
        self.stage1_action_dim = self.item_dim + 1
        self.stage1_action_layer = nn.Linear(self.state_dim, self.stage1_action_dim)
        # stage 2 action is the 2 layer scorer parameters
#         self.dnn_dims = [self.state_dim] + args.stage2_scorer_hidden_dims + [1]
        self.stage2_hidden_dim = args.stage2_scorer_hidden_dim
        self.stage2_action_dim = (self.item_dim + 2) * (self.stage2_hidden_dim + 1)
#         self.dnn_dims = [self.state_dim, args.stage2_scorer_hidden_dim, 1]
#         self.stage2_action_dim = sum([(self.dnn_dims[i]+1) * self.dnn_dims[i+1] for i in range(len(self.dnn_dims)-1)]) \
#                                         + self.item_dim + 1
        self.stage2_action_layer = nn.Linear(self.state_dim, self.stage2_action_dim)
#         self.stage2_action_layer = DNN(self.state_dim + self.stage1_action_dim, [128], self.stage2_action_dim, 
#                                        dropout_rate = args.stage2_dropout_rate, do_batch_norm = False)
        self.stage2_scorer_norm = nn.LayerNorm(args.stage2_scorer_hidden_dim)

    def get_action_dim(self):
        return {'total': self.stage1_action_dim + self.stage2_action_dim, 
                'separate': [self.stage1_action_dim, self.stage2_action_dim]}
        
    def score(self, action_emb, item_emb, stage = 1):
        '''
        @input:
        - action_emb: (B, (i_dim+1))
        - item_emb: (B, L, i_dim) or (1, L, i_dim)
        
        @output:
        - scores: (B, L)
        '''
        if stage == 1:
            return linear_scorer(action_emb, item_emb, self.item_dim)
        else:
            return wide_and_deep_scorer(action_emb, item_emb, self.item_dim, self.stage2_hidden_dim, 
                                        self.stage2_scorer_norm, self.dropout_rate)
#         # scoring model parameters
#         # (B, 1, i_dim)
#         fc_weight = action_emb[:, :self.item_dim].view(-1, 1, self.item_dim) # * 2 / math.sqrt(self.item_dim)
#         # (B, 1)
#         fc_bias = action_emb[:,-1].view(-1, 1)
        
#         # forward
#         output = torch.sum(fc_weight * item_emb, dim = -1) + fc_bias
#         # (B, L)
#         return torch.softmax(output, dim = -1)

    def forward(self, feed_dict):
        '''
        @input:
        - feed_dict: {'user_profile': (B, user_dim), 
                    'history_features': (B, H, item_dim, 
                    'candidate_features': (B, L, item_dim) or (1, L, item_dim), 
                    'stage1_noise_var': scalar, 'stage2_noise_var': scalar}
        @model:
        - feed_dict --(state_encoder)--> state_emb (B,1,state_dim)
        - state_emb --(action1_layer)--> stage1_action_emb (B,action1_dim)
        - state_emb --(action2_layer)--> stage2_action_emb (B,action2_dim)
        
        @output:
        - out_dict: {"action1_emb": (B, action1_dim), 
                        "action2_emb": (B, action2_dim),
                        "state_emb": (B,f_dim),
                        "reg": scalar}
        '''
        # user embedding (B,f_dim)
        user_state = self.state_encoder(feed_dict)['state_emb']
        B = user_state.shape[0]
        # action embedding (B,action_dim)
        stage1_action_emb = self.stage1_action_layer(user_state).view(B, self.stage1_action_dim)
        stage2_action_emb = self.stage2_action_layer(user_state).view(B, self.stage2_action_dim)
        # sampling noise of action embedding
        if 'stage1_noise_var' in feed_dict:
            stage1_action_emb = stage1_action_emb + torch.randn_like(stage1_action_emb) * feed_dict['stage1_noise_var']
        if 'stage2_noise_var' in feed_dict:
            stage2_action_emb = stage2_action_emb + torch.randn_like(stage2_action_emb) * feed_dict['stage2_noise_var']
        # regularization terms
#         reg = get_regularization(self.state_encoder, self.stage1_action_layer, self.stage2_action_layer)
        reg = 0
        # output
        out_dict = {'action1_emb': stage1_action_emb, 
                    'action2_emb': stage2_action_emb, 
                    'state_emb': user_state.view(B,-1),
                    'reg': reg}
        return out_dict