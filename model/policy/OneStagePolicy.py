import torch.nn.functional as F
import torch.nn as nn
import torch
import math

from model.components import DNN
from utils import get_regularization

class BaseStateEncoder(nn.Module):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - state_encoder_feature_dim
        - state_encoder_attn_n_head
        - state_encoder_hidden_dims
        - state_encoder_dropout_rate
        '''
        parser.add_argument('--state_encoder_feature_dim', type=int, default=32, 
                            help='dimension size for state')
        parser.add_argument('--state_encoder_attn_n_head', type=int, default=4, 
                            help='dimension size for all features')
        parser.add_argument('--state_encoder_hidden_dims', type=int, nargs='+', default=[128], 
                            help='specificy a list of k for top-k performance')
        parser.add_argument('--state_encoder_dropout_rate', type=float, default=0.1, 
                            help='dropout rate in deep layers')
        return parser
    
    def __init__(self, args, environment):
        super().__init__()
        # action space
        self.item_dim = environment.action_space['item_feature'][1]
        # observation space
        user_profile_info = environment.observation_space['user_profile']
        self.user_dim = user_profile_info[1]
        # state space
        self.f_dim = args.state_encoder_feature_dim
        self.state_dim = self.f_dim # + self.user_dim
        # policy network modules
        self.user_profile_encoder = DNN(self.user_dim, args.state_encoder_hidden_dims, self.f_dim, 
                                        dropout_rate = args.state_encoder_dropout_rate, do_batch_norm = True)
        self.item_emb_layer = nn.Linear(self.item_dim, self.f_dim)
        self.seq_user_attn_layer = nn.MultiheadAttention(self.f_dim, args.state_encoder_attn_n_head, batch_first = True)
        self.state_linear = nn.Linear(self.f_dim + self.user_dim, self.state_dim)
        self.state_norm = nn.LayerNorm([self.state_dim])
        # To be implemented: action modules
        
        
    def forward(self, feed_dict):
        '''
        @input:
        - feed_dict: {'user_profile': (B, user_dim), 
                    'history_features': (B, H, item_dim, 
                    'candidate_features': (B, L, item_dim) or (1, L, item_dim)}
        @model:
        - user_profile --> user_emb (B,1,f_dim)
        - history_items --> history_item_emb (B,H,f_dim)
        - (Q:user_emb, K&V:history_item_emb) --(multi-head attn)--> user_state (B,1,f_dim)
        - user_state --> action_prob (B,n_item)
        @output:
        - out_dict: {"action_emb": (B,action_dim), 
                     "state_emb": (B,f_dim),
                     "reg": scalar,
                     "action_prob": (B,L), include probability score when candidate_features are given}
        '''
        # user embedding (B,1,f_dim)
        user_emb = self.user_profile_encoder(feed_dict['user_profile']).view(-1,1,self.f_dim)
        B = user_emb.shape[0]
        # history embedding (B,H,f_dim)
        history_item_emb = self.item_emb_layer(feed_dict['history_features'])
        # cross attention, encoded history is (B,1,f_dim)
        user_state, attn_weight = self.seq_user_attn_layer(user_emb, history_item_emb, history_item_emb)
        # (B, 2*f_dim)
#         user_state = torch.cat((user_state.view(B, self.f_dim), user_emb.view(B, self.f_dim)), dim = -1)
#         user_state = torch.sigmoid(self.state_linear(user_state))
        user_state = self.state_linear(torch.cat((user_state.view(B, self.f_dim),
                                                  feed_dict['user_profile'].view(B,self.user_dim)), dim = -1))
        user_state = self.state_norm(user_state)
#         user_state = torch.sigmoid(user_state)
#         reg = get_regularization(self.user_profile_encoder, self.item_emb_layer, 
#                                  self.seq_user_attn_layer, self.action_layer)
        return {'state_emb': user_state}
    
class OneStagePolicy(nn.Module):
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
#         parser.add_argument('--policy_actionnet_range', type=float, default=1.0, 
#                             help='value range of the action net')
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
        # To be implemented: self.item_dim, self.action_layer
        
    def score(self, action_emb, item_emb, do_softmax = True):
        '''
        @input:
        - action_emb: (B, (i_dim+1))
        - item_emb: (B, L, i_dim) or (1, L, i_dim)
        @output:
        - scores: (B, L)
        '''
        pass

    def forward(self, feed_dict):
        '''
        @input:
        - feed_dict: {'user_profile': (B, user_dim), 
                    'history_features': (B, H, item_dim, 
                    'candidate_features': (B, L, item_dim) or (1, L, item_dim)}
        @model:
        - user_profile --> user_emb (B,1,f_dim)
        - history_items --> history_item_emb (B,H,f_dim)
        - (Q:user_emb, K&V:history_item_emb) --(multi-head attn)--> user_state (B,1,f_dim)
        - user_state --> action_prob (B,n_item)
        @output:
        - out_dict: {"action_emb": (B,action_dim), 
                     "state_emb": (B,f_dim),
                     "reg": scalar,
                     "action_prob": (B,L), include probability score when candidate_features are given}
        '''
        # user embedding (B,1,f_dim)
        user_state = self.state_encoder(feed_dict)['state_emb']
        B = user_state.shape[0]
        # action embedding (B,action_dim)
        action_emb = self.action_layer(user_state).view(B, self.action_dim)
        # regularization terms
        reg = get_regularization(self.state_encoder, self.action_layer)
        # output
        out_dict = {'action_emb': action_emb, 
                    'state_emb': user_state.view(B,-1),
                    'reg': reg}
#         if 'candidate_features' in feed_dict:
#             # action prob (B,L)
#             action_prob = self.score(action_emb, feed_dict['candidate_features'], feed_dict['do_softmax'])
#             out_dict['action_prob'] = action_prob
#             out_dict['candidate_ids'] = feed_dict['candidate_ids']
        return out_dict