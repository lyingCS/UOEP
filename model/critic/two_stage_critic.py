
import torch.nn.functional as F
import torch.nn as nn
import torch

from model.components import DNN
from utils import get_regularization

# from model.agents.critic import GeneralCritic

class JointActionCritic(nn.Module):
    '''
    Standard critic that evaluate joint actions from multiple agents
    '''
    
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - critic_hidden_dims
        - critic_dropout_rate
        '''
        parser.add_argument('--critic_hidden_dims', type=int, nargs='+', default=[256, 64], 
                            help='hidden dimensions of critic network')
        parser.add_argument('--critic_dropout_rate', type=float, default=0.2, 
                            help='dropout rate in deep layers')
        return parser
    
    def __init__(self, args, environment, policy):
        '''
        self.state_dim = policy.state_dim
        self.action_dim = policy.action_dim
        self.action_map = nn.Linear(policy.action_dim, policy.state_dim)
        self.net = DNN(self.state_dim * 2, args.critic_hidden_dims, 1, 
                       dropout_rate = args.critic_dropout_rate, do_batch_norm = False)
        '''
        super().__init__()
        self.state_dim = policy.state_dim
        action_dim_info = policy.get_action_dim()
        self.total_action_dim = action_dim_info['total']
        self.separate_action_dims = action_dim_info['separate']
        self.net = DNN(self.state_dim + self.total_action_dim, args.critic_hidden_dims, 1, 
                        dropout_rate = args.critic_dropout_rate, do_batch_norm = False)
        
    def forward(self, feed_dict):
        '''
        @input:
        - feed_dict: {'state_emb': (B, state_dim), 'action_emb': (B, action_dim)}
        '''
        state_emb = feed_dict['state_emb']
        # [(B, action_dim)]
        action_emb_list = [feed_dict['action_emb'][i].view(-1, d) for i,d in enumerate(self.separate_action_dims)]
        # (B, total_action_dim)
        action_emb = torch.cat(action_emb_list, dim = -1)
        # (B,)
        Q = self.net(torch.cat((state_emb, action_emb), dim = -1)).view(-1)
#         reg = get_regularization(self.net)
        reg = 0
        return {'q': Q, 'reg': reg}
    
    
class TwoStageCritic(JointActionCritic):
    '''
    Standard critic that evaluate joint actions from multiple agents
    '''
    
    def __init__(self, args, environment, policy):
        '''
        self.state_dim = policy.state_dim
        self.action_dim = policy.action_dim
        self.action_map = nn.Linear(policy.action_dim, policy.state_dim)
        self.net = DNN(self.state_dim * 2, args.critic_hidden_dims, 1, 
                       dropout_rate = args.critic_dropout_rate, do_batch_norm = False)
        '''
        super().__init__(args, environment, policy)
        self.action2_dim = self.separate_action_dims[1]
        self.separate_action_dims[1] = self.separate_action_dims[0]
        self.action2_mapping = nn.Linear(self.action2_dim, self.separate_action_dims[0])
        self.action2_norm = nn.LayerNorm(self.separate_action_dims[0])
        
        self.total_action_dim = sum(self.separate_action_dims)
        self.net = DNN(self.state_dim + self.total_action_dim, args.critic_hidden_dims, 1, 
                        dropout_rate = args.critic_dropout_rate, do_batch_norm = True)
        
    def forward(self, feed_dict):
        '''
        @input:
        - feed_dict: {'state_emb': (B, state_dim), 'action1_emb': (B, action_dim), 'action2_emb': (B, action_dim)}
        '''
        feed_dict['action_emb'] = [feed_dict['action1_emb'], 
                                   self.action2_norm(self.action2_mapping(feed_dict['action2_emb']))]
        return super().forward(feed_dict)
    
    