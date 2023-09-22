import numpy as np
import pandas as pd
from tqdm import tqdm

from reader.BaseReader import BaseReader
from utils import padding_and_clip, get_frequency_dict_of_seq_feature

class KRDataReader(BaseReader):
    
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
        parser.add_argument('--user_meta_file', type=str, required=True, 
                            help='item raw feature file_path')
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
        print("init kr reader")
        self.max_seq_len = args.max_seq_len
        super().__init__(args)


    def _read_data(self, args):
        # read data_file
        super()._read_data(args)
        print("Load item meta data")
        # self.item_meta = pd.read_table(args.item_meta_file, sep = args.meta_data_separator, engine = 'python')
        self.item_meta = np.load(args.item_meta_file)
        self.user_meta = np.load(args.user_meta_file)
        self.item_vec_size = len(self.item_meta[0])
        self.user_vec_size = len(self.user_meta[0])
        # print(self.item_meta.shape)
        # self.item_meta = np.concatenate((np.zeros_like(self.item_meta[0]).reshape(1, -1), self.item_meta), axis=0)
        # self.user_meta = np.concatenate((np.zeros_like(self.user_meta[0]).reshape(1, -1), self.user_meta), axis=0)

        # rating_data = pd.read_table(args.rating_data_file, sep = "::", header = None, skiprows = 1,
        #                     names = ["UserID", "ItemID", "Rating", "Timestamp"], engine = 'python')
        # rating_data = rating_data.sort_values(by=['UserID','Timestamp'])

        # self.users = list(rating_data.UserID.unique())
        # self.user_vocab = {uid: idx+1 for idx,uid in enumerate(self.users)} # padding = 0

        # self.items = list(rating_data.ItemID.unique())
        # self.item_vocab = {iid: idx+1 for idx,iid in enumerate(self.items)} # padding = 0
        # self.item_vocab[0]=0
#         self.portrait_vocab = get_vocab_of_seq_feature(self.data['train'], 'user_protrait')
        self.portrait_len = len(self.user_meta[0])
    
    ###########################
    #        Iterator         #
    ###########################
        
    def __getitem__(self, idx):
        '''
        train batch after collate:
        {
        'resp': 2, 
        'user_UserID': (B,) 
        'user_XXX': (B,feature_size)
        'item_ItemID': (B,)
        'item_XXX': (B,feature_size)
        'negi_ItemID': (B,n_neg) 
        'negi_XXX': (B,n_neg,feature_size) 
        }
        '''
        user_ID, slate_of_items, user_clicks, user_click_history, sequence_id = self.data[self.phase].iloc[idx]
        user_profile = self.user_meta[user_ID]


        exposure = eval(slate_of_items)
        # exposure = list(map(lambda x: self.item_vocab[x], exposure))
        # exposure = padding_and_clip(exposure, 10) # test
        history = eval(user_click_history)
        # history = list(map(lambda x: self.item_vocab[x], history))
        hist_length = len(history)
        history = padding_and_clip(history, self.max_seq_len)
        feedback = eval(user_clicks)
        record = {
            'timestamp': int(1),
            # 'timestamp': int(timestamp),
            'exposure': np.array(exposure).astype(int), 
            'exposure_features': self.get_item_list_meta(exposure).astype(float),
            'feedback': np.array(feedback).astype(float),
            'history': np.array(history).astype(int),
            'history_features': self.get_item_list_meta(history).astype(float),
            'history_length': int(min(hist_length, self.max_seq_len)),
            'user_profile': np.array(user_profile)
            # 'user_profile': np.array(user_ID).astype(int)
        }
        return record

    def get_row_data(self, pick_rows):
        record = {
            'timestamp': [],
            # 'timestamp': int(timestamp),
            'exposure': [],
            'exposure_features': [],
            'feedback': [],
            'history': [],
            'history_features': [],
            'history_length': [],
            'user_profile': []
            # 'user_profile': np.array(user_ID).astype(int)
        }
        for row in pick_rows:
            user_ID, slate_of_items, user_clicks, user_click_history, sequence_id = self.data[self.phase].iloc[row]
            user_profile = self.user_meta[user_ID]


            exposure = eval(slate_of_items)
            # exposure = list(map(lambda x: self.item_vocab[x], exposure))
            # exposure = padding_and_clip(exposure, 10) # test
            history = eval(user_click_history)
            # history = list(map(lambda x: self.item_vocab[x], history))
            hist_length = len(history)
            history = padding_and_clip(history, self.max_seq_len)
            feedback = eval(user_clicks)
            record['timestamp'].append(int(1))
            # 'timestamp': int(timestamp),
            record['exposure'].append(np.array(exposure).astype(int))
            record['exposure_features'].append(self.get_item_list_meta(exposure).astype(float))
            record['feedback'].append(np.array(feedback).astype(float))
            record['history'].append(np.array(history).astype(int))
            record['history_features'].append(self.get_item_list_meta(history).astype(float))
            record['history_length'].append(int(min(hist_length, self.max_seq_len)))
            record['user_profile'].append(np.array(user_profile))
        for key in record.keys():
            record[key] = np.array(record[key])
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
                features.append(self.item_meta[iid-1])
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
