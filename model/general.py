import torch
import torch.nn as nn
import numpy as np

from utils import get_regularization

# def init_weights(module, aux = {}, m_type = "embedding"):
#     sigma = 1.0 / np.sqrt(aux["dim"]) if "dim" in aux else 0.01
#     if m_type == "embedding":
#         nn.init.normal_(module.weight, mean = 0.0, std = sigma)
#     elif m_type == "bias":
#         nn.init.normal_(module.weight, mean = 0.0, std = 0.01)
#     elif m_type == 'linear':
#         nn.init.normal_(module.weight, mean=0.0, std=sigma)
#         if module.bias is not None:
#             nn.init.normal_(module.bias, mean=0.0, std=sigma)

# def get_l2_regularization(module, m_type = "mlp"):
#     if m_type == "mlp":
#         reg = torch.tensor(0., requires_grad=True).to(next(module[0].parameters()).device)
#         for subModule in module:
#             reg = reg + torch.sum(subModule.weight ** 2)
#         return reg

class BaseModel(nn.Module):

    #############################
    #     Optional Overwrite    #
    #############################
    
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--model_path', type=str, default='',
                            help='Model save path.')
        parser.add_argument('--loss', type=str, default='',
                            help='loss type')
        parser.add_argument('--l2_coef', type=float, default=0.,
                            help='coefficient of regularization term')
        return parser
        
    def log(self):
        self.reader.log()
        print("Model params")
        print("\tmodel_path = " + str(self.model_path))
        print("\tloss_type = " + str(self.loss_type))
        print("\tl2_coef = " + str(self.l2_coef))
        print("\tdevice = " + str(self.device))
    
    def __init__(self, args, reader, device):
        super(BaseModel, self).__init__()
        self.display_name = "BaseModel"
        self.reader = reader
        self.model_path = args.model_path
        self.loss_type = args.loss
        self.l2_coef = args.l2_coef
        self.no_reg = 0. < args.l2_coef < 1.
        self.device = device

        self._define_params(args, reader)
        
#         self.MSE = nn.MSELoss(reduction = 'none')
#         self.BCE = nn.BCELoss(reduction = 'none')
        self.sigmoid = nn.Sigmoid()
    
    def get_regularization(self, *modules):
        return get_regularization(*modules)
    
    def show_params(self):
        print(f"All parameters for {self.display_name}========================")
        idx = 0
        all_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                # try:
                param_shape = list(param.size())
                print(" var {:3}: {:15} {}".format(idx, str(param_shape), name))
                num_params = 1
                if (len(param_shape) > 1):
                    for p in param_shape:
                        if (p > 0):
                            num_params = num_params * int(p)
                    all_params.append(num_params)
                elif len(param_shape) == 1:
                    all_params.append(param_shape[0])
                else:
                    all_params.append(1)
                idx += 1
        num_fixed_params = np.sum(all_params)
        idx = 0
        all_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                # try:
                param_shape = list(param.size())
                print(" var {:3}: {:15} {}".format(idx, str(param_shape), name))
                num_params = 1
                if (len(param_shape) > 1):
                    for p in param_shape:
                        if (p > 0):
                            num_params = num_params * int(p)
                    all_params.append(num_params)
                elif len(param_shape) == 1:
                    all_params.append(param_shape[0])
                else:
                    all_params.append(1)
                idx += 1
        num_params = np.sum(all_params)
        print("Total number of trainable params {}".format(num_params))
        print("Total number of fixed params {}".format(num_fixed_params))
    
    def do_forward_and_loss(self, feed_dict: dict) -> dict:
        '''
        Called during training
        '''
        out_dict = self.forward(feed_dict)
        loss = self.get_loss(feed_dict, out_dict)
        out_dict["loss"] = loss
        return out_dict
    
    def forward(self, feed_dict: dict, return_prob = True) -> dict:
        '''
        Called during evaluation or prediction
        '''
        out_dict = self.get_forward(feed_dict)
        if return_prob:
            out_dict["probs"] = nn.Sigmoid()(out_dict["preds"])
        return out_dict
        
    def wrap_batch(self, batch):
        '''
        Build feed_dict from batch data and move data to self.device
        '''
        for k,val in batch.items():
            if type(val).__module__ == np.__name__:
                batch[k] = torch.from_numpy(val)
            elif torch.is_tensor(val):
                batch[k] = val
            elif type(val) is list:
                batch[k] = torch.tensor(val)
            else:
                continue
            if batch[k].type() == "torch.DoubleTensor":
                batch[k] = batch[k].float()
            batch[k] = batch[k].to(self.device)
        return batch
    
    def save_checkpoint(self):
        torch.save({
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, self.model_path + ".checkpoint")
        print("Model (checkpoint) saved to " + self.model_path)
    
    def load_from_checkpoint(self, model_path, with_optimizer = True):
        print("Load (checkpoint) from " + model_path + ".checkpoint")
        checkpoint = torch.load(model_path + ".checkpoint", map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])
        if with_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.model_path = model_path
    
    def actions_before_train(self, info):  # e.g. initialization
        pass

    def actions_after_train(self, info):  # e.g. compression
        pass
    
    def actions_before_epoch(self, info): # e.g. expectation update
        pass
    
    def actions_after_epoch(self, info): # e.g. prunning
        pass
    
    #############################
    #   Require Implementation  #
    #############################
    
    
    def _define_params(self, args, reader):
        pass
    
    def get_forward(self, feed_dict: dict) -> dict:
        pass
    
    def get_loss(self, feed_dict: dict, out_dict: dict):
        pass