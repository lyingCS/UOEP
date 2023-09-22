from argparse import Namespace

from env.reward import *

class BaseRLEnvironment():
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - env_path
        '''
        parser.add_argument('--env_path', type=str, required=True)
#         parser.add_argument('--rl_batch_size', type=int, default=1, help='batch size for RL')
        parser.add_argument('--reward_func', type=str, default='mean_with_cost', help='reward function name')
        parser.add_argument('--max_step_per_episode', type=int, default=100, help='max number of iteration allowed in each episode')
        parser.add_argument('--initial_temper', type=int, required=100, help='initial temper of users')
        return parser
    
    
    def __init__(self, args):
        super().__init__()
        self.env_path = args.env_path
#         self.rl_batch_size = args.rl_batch_size
        self.reward_func = eval(args.reward_func)
        self.max_step_per_episode = args.max_step_per_episode
        self.initial_temper = args.initial_temper
        
    def reset(self, params):
        pass
        
    def step(self, action):
        pass
    