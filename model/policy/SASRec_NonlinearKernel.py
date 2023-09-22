import torch.nn.functional as F
import torch.nn as nn
import torch
import math

from model.components import DNN
from model.policy.SASRec import SASRec
from model.score_func import dot_scorer
    
class SASRec_NonlinearKernel(SASRec):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        '''
        parser = SASRec.parse_model_args(parser)
        parser.add_argument('--item_kernel_hidden', type=int, nargs='+', default=[128], 
                            help='hidden dims of item kernel')
        return parser
    
    def __init__(self, args, environment):
        '''
        action_space = {'item_id': ('nominal', stats['n_item']), 
                        'item_feature': ('continuous', stats['item_vec_size'], 'normal')}
        observation_space = {'user_profile': ('continuous', stats['user_portrait_len'], 'positive']), 
                            'history': ('sequence', stats['max_seq_len'], ('continuous', stats['item_vec_size']))}
        '''
        super().__init__(args, environment)
        self.item_map = DNN(self.item_dim, args.item_kernel_hidden, self.d_model, 
                            dropout_rate = args.critic_dropout_rate, do_batch_norm = True)
        
    def score(self, action_emb, item_emb, do_softmax = True):
        '''
        @input:
        - action_emb: (B, (i_dim))
        - item_emb: (B, L, i_dim) or (1, L, i_dim)
        @output:
        - scores: (B, L)
        '''
        item_set_size = item_emb.shape[1]
        item_emb = self.item_map(item_emb).view(-1,item_set_size, self.d_model)
        output = dot_scorer(action_emb, item_emb, self.d_model)
        if do_softmax:
            return torch.softmax(output, dim = -1)
        else:
            return output
        