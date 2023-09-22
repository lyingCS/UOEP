from matplotlib.pyplot import axes, axis
from model.general import BaseModel
from model.components import DNN
import torch
import torch.nn as nn

class KRUserResponse(BaseModel):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - feature_dim
        - attn_n_head
        - hidden_dims
        - dropout_rate
        - batch_norm
        - from BaseModel:
            - model_path
            - loss
            - l2_coef
        '''
        parser = BaseModel.parse_model_args(parser)
        
        parser.add_argument('--feature_dim', type=int, default=32, 
                            help='dimension size for all features')
        parser.add_argument('--attn_n_head', type=int, default=4, 
                            help='dimension size for all features')
        parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128], 
                            help='specificy a list of k for top-k performance')
        parser.add_argument('--dropout_rate', type=float, default=0.2, 
                            help='dropout rate in deep layers')
        return parser
    
    def log(self):
        super().log()
        print("\tencoding_dim = " + str(self.feature_dim))
        print("\titem_input_dim = " + str(self.feature_dim))
        print("\tuser_input_dim = " + str(self.feature_dim))
        
    def __init__(self, args, reader, device):
        super().__init__(args, reader, device)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction = 'none')

    def _define_params(self, args, reader):
        stats = reader.get_statistics()
        print(stats)
        self.portrait_len = stats['user_portrait_len']
        self.item_dim = stats['item_vec_size']
        self.feature_dim = args.feature_dim
        # self.uEmb = nn.Embedding.from_pretrained(torch.FloatTensor(self.reader.user_meta), freeze=False)
        # self.iEmb = nn.Embedding.from_pretrained(torch.FloatTensor(self.reader.item_meta), freeze=False)

        self.uEmb = nn.Embedding(986, 32)
        self.iEmb = nn.Embedding(11644, 32)
        self.concat_layer = nn.Linear(args.feature_dim * 2, args.feature_dim)
        # self.uEmb = nn.Embedding.load_state_dict()
        # portrait embedding
        self.portrait_encoding_layer = DNN(self.portrait_len, args.hidden_dims, args.feature_dim, 
                                           dropout_rate = args.dropout_rate, do_batch_norm = False)
        # item_emb
        self.item_emb_layer = nn.Linear(self.item_dim, args.feature_dim)
        # user history encoder
        self.seq_self_attn_layer = nn.MultiheadAttention(args.feature_dim, args.attn_n_head, batch_first = True)
        self.seq_user_attn_layer = nn.MultiheadAttention(args.feature_dim, args.attn_n_head, batch_first = True)
    
    # def get_forward(self, feed_dict: dict):
        # # print(feed_dict['user_profile'])
        # user_emb = self.uEmb(feed_dict['user_profile'])
        # # print(user_emb.shape)
        # user_emb = self.portrait_encoding_layer(user_emb).view(-1,1,self.feature_dim)

        # item_emb = self.iEmb(feed_dict['history'])
        # # user embedding (B,1,f_dim)
        # # user_emb = self.portrait_encoding_layer(feed_dict['user_profile']).view(-1,1,self.feature_dim) 
        # # history embedding (B,H,f_dim)

        # history_item_emb = self.item_emb_layer(item_emb)

        # # history_item_emb = self.item_emb_layer(feed_dict['history_features'])



        # # sequence self attention, encoded sequence is (B,H,f_dim)
        # seq_encoding, attn_weight = self.seq_self_attn_layer(history_item_emb, history_item_emb, history_item_emb)
        # # cross attention, encoded history is (B,1,f_dim)
        # user_interest, attn_weight = self.seq_user_attn_layer(user_emb, seq_encoding, seq_encoding)
        # # rec item embedding (B,L,f_dim)
        # # print(user_interest.shape)
        # # print(user_emb.shape)
        # user_interest = torch.concat([user_interest, user_emb], axis=-1) # waiting 
        # user_interest = self.concat_layer(user_interest)                            # waiting 

        # # self.iEmb(feed_dict['exposure'])

        # exposure_item_emb = self.item_emb_layer(self.iEmb(feed_dict['exposure']))

        # # exposure_item_emb = self.item_emb_layer(feed_dict['exposure_features']) #.view(-1,len(feed_dict['exposure_features']),self.feature_dim) 
        # # rec item attention score (B,L)


        # score = torch.sum(exposure_item_emb * user_interest, dim = -1)
        # # regularization terms
        # reg = self.get_regularization(self.uEmb, self.iEmb, self.portrait_encoding_layer, self.item_emb_layer, 
        #                               self.seq_user_attn_layer, self.seq_self_attn_layer)
        # return {'preds': score, 'reg': reg}

    def get_forward(self, feed_dict: dict):
        # print(feed_dict['user_profile'])
        # user_emb = self.uEmb(feed_dict['user_profile'])
        # print(user_emb.shape)
        user_emb = self.portrait_encoding_layer(feed_dict['user_profile']).view(-1,1,self.feature_dim)

        # item_emb = self.iEmb(feed_dict['history'])
        # user embedding (B,1,f_dim)
        # user_emb = self.portrait_encoding_layer(feed_dict['user_profile']).view(-1,1,self.feature_dim) 
        # history embedding (B,H,f_dim)

        # history_item_emb = self.item_emb_layer(feed_dict['history'])

        history_item_emb = self.item_emb_layer(feed_dict['history_features'])



        # sequence self attention, encoded sequence is (B,H,f_dim)
        seq_encoding, attn_weight = self.seq_self_attn_layer(history_item_emb, history_item_emb, history_item_emb)
        # cross attention, encoded history is (B,1,f_dim)
        user_interest, attn_weight = self.seq_user_attn_layer(user_emb, seq_encoding, seq_encoding)
        # rec item embedding (B,L,f_dim)
        user_interest = torch.concat([user_interest, user_emb], axis=-1) # waiting 
        user_interest = self.concat_layer(user_interest)                            # waiting 
        exposure_item_emb = self.item_emb_layer(feed_dict['exposure_features'])

        # exposure_item_emb = self.item_emb_layer(feed_dict['exposure_features']) #.view(-1,len(feed_dict['exposure_features']),self.feature_dim) 
        # rec item attention score (B,L)


        score = torch.sum(exposure_item_emb * user_interest, dim = -1)
        # regularization terms
        reg = self.get_regularization(self.uEmb, self.iEmb, self.portrait_encoding_layer, self.item_emb_layer, 
                                      self.seq_user_attn_layer, self.seq_self_attn_layer)
        return {'preds': score, 'reg': reg}
    
    def get_loss(self, feed_dict: dict, out_dict: dict):
        """
        @input:
        - feed_dict: {...}
        - out_dict: {"preds":, "reg":}
        
        Loss terms implemented:
        - BCE
        """
        
        preds, reg = out_dict["preds"].view(-1), out_dict["reg"] # (B,L), scalar
        target = feed_dict['feedback'].view(-1).to(torch.float) # (B,L)
        loss = torch.mean(self.bce_loss(self.sigmoid(preds), target))
        loss = loss + self.l2_coef * reg
        return loss
    
    