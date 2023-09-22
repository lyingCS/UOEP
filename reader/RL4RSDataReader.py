import numpy as np
import pandas as pd
from tqdm import tqdm

from reader.BaseReader import BaseReader
from utils import padding_and_clip, get_frequency_dict_of_seq_feature

class RL4RSDataReader(BaseReader):
    
    @staticmethod
    def parse_data_args(parser):
        '''
        args:
        - from BaseReader:
            - train_file
            - val_file
            - test_file
            - n_worker
        '''
        parser = BaseReader.parse_data_args(parser)
        parser.add_argument('--item_meta_file', type=str, required=True, 
                            help='item raw feature file_path')
        parser.add_argument('--max_seq_len', type=int, default=50, 
                            help='item raw feature file_path')
        parser.add_argument('--meta_data_separator', type=str, default='\t',
                            help='separator of item_meta file')
        return parser
        
    def log(self):
        super().log()
        
    def __init__(self, args):
        '''
        - from BaseReader:
            - phase
            - data: will add Position column
        '''
        self.max_seq_len = args.max_seq_len
        super().__init__(args)
        
    def _read_data(self, args):
        # read data_file
        super()._read_data(args)
        print("Load item meta data")
        self.item_meta = pd.read_table(args.item_meta_file, sep = args.meta_data_separator, engine = 'python')
        self.item_vec_size = len(eval(self.item_meta.iloc[0]['item_vec']))
#         self.portrait_vocab = get_vocab_of_seq_feature(self.data['train'], 'user_protrait')
        self.portrait_len = len(eval(self.data['train']['user_protrait'][0]))
    
    ###########################
    #        Iterator         #
    ###########################
        
    def __getitem__(self, idx):
        '''
        train batch after collate:
        {
        'timestamp': (B,), 
        'exposure': (B,K) 
        'exposure_features': (B,K,item_dim)
        'feedback': (B,K)
        'history': (B,H)
        'history_features': (B,H,item_dim) 
        'history_features': (B,) 
        'user_profile': (B,user_dim) 
        }
        '''
        timestamp, _, __, exposure, feedback, history, portrait, ___, ____ = self.data[self.phase].iloc[idx]
        exposure = eval(exposure)
        history = eval(f"[{history}]")
        hist_length = int(min(len(history), self.max_seq_len))
        history = padding_and_clip(history, self.max_seq_len)
        feedback = eval(feedback)
        record = {
            'timestamp': int(timestamp),
            'exposure': np.array(exposure).astype(int), 
            'exposure_features': self.get_item_list_meta(exposure).astype(float),
            'feedback': np.array(feedback).astype(float),
            'history': np.array(history).astype(int),
            'history_features': self.get_item_list_meta(history).astype(float),
            'history_length': hist_length,
            'history_mask': np.array(padding_and_clip([1]*hist_length, self.max_seq_len)),
            'user_profile': np.log(np.array(eval(portrait)) + 1)
        }
        return record
        
    def get_item_list_meta(self, iid_list, from_idx = False):
        '''
        @input:
        - iid_list: item id list
        @output:
        - meta_data: {field_name: (B,feature_size)}
        '''
        features = []
        for iid in iid_list:
            if iid == 0:
                features.append([0]*self.item_vec_size)
            else:
                features.append(eval(self.item_meta.iloc[iid-1]['item_vec']))
        return np.array(features)
    
    def get_statistics(self):
        '''
        - n_user
        - n_item
        - s_parsity
        - from BaseReader:
            - length
            - fields
        '''
        stats = super().get_statistics()
        stats["n_item"] = len(self.item_meta)
        stats["item_vec_size"] = self.item_vec_size
        stats["user_portrait_len"] = self.portrait_len
        stats["max_seq_len"] = self.max_seq_len
        stats["n_feedback"] = 2
        return stats