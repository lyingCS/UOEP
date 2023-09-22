from tqdm import tqdm
import random
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import os

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def check_folder_exist(fpath):
    if os.path.exists(fpath):
        print("dir \"" + fpath + "\" existed")
    else:
        try:
            os.mkdir(fpath)
        except:
            print("error when creating \"" + fpath + "\"") 
            
def setup_path(fpath, is_dir = True):
    dirs = [p for p in fpath.split("/")]
    curP = ""
    dirs = dirs[:-1] if not is_dir else dirs
    for p in dirs:
        curP += p
        check_folder_exist(curP)
        curP += "/"

###########################################
#              Data Related               #
###########################################

def padding_and_clip(sequence, max_len, padding_direction = 'left'):
    if len(sequence) < max_len:
        sequence = [0] * (max_len - len(sequence)) + sequence if padding_direction == 'left' else sequence + [0] * (max_len - len(sequence))
    sequence = sequence[-max_len:] if padding_direction == 'left' else sequence[:max_len]
    return sequence
    
def get_vocab_of_seq_feature(df, seq_feature_key):
    print(f"Get vocab for feature({seq_feature_key})")
    vocab = {}
    for i,row in tqdm(df.iterrows()):
        value_list = row[seq_feature_key].split(',')
        for v in value_list:
            if v not in vocab:
                vocab[v] = len(vocab) + 1
    return vocab

def get_frequency_dict_of_seq_feature(column):
    len_dict = {}
    for seq in tqdm(column):
        L = len(seq.split(','))
        if L not in len_dict:
            len_dict[L] = 1
        else:
            len_dict[L] += 1
    return len_dict


def show_batch(batch):
    for k, batch in batch.items():
        if torch.is_tensor(batch):
            print(f"{k}: size {batch.shape}, \n\tfirst 5 {batch[:5]}")
        else:
            print(f"{k}: {batch}")
            

def wrap_batch(batch, device):
    '''
    Build feed_dict from batch data and move data to device
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
        batch[k] = batch[k].to(device)
    return batch


############################################
#              Model Related               #
############################################

def init_weights(m):
    if 'Linear' in str(type(m)):
#         nn.init.normal_(m.weight, mean=0.0, std=0.01)
        nn.init.xavier_normal_(m.weight, gain=1.)
        if m.bias is not None:
            nn.init.normal_(m.bias, mean=0.0, std=0.01)
    elif 'Embedding' in str(type(m)):
#         nn.init.normal_(m.weight, mean=0.0, std=0.01)
        nn.init.xavier_normal_(m.weight, gain=1.0)
        print("embedding: " + str(m.weight.data))
        with torch.no_grad():
            m.weight[m.padding_idx].fill_(0.)
    elif 'ModuleDict' in str(type(m)):
        for param in module.values():
            nn.init.xavier_normal_(param.weight, gain=1.)
            with torch.no_grad():
                param.weight[param.padding_idx].fill_(0.)
                
                
def get_regularization(*modules):
    reg = 0
    for m in modules:
        for p in m.parameters():
            reg = torch.mean(p * p) + reg
    return reg


def torch_cat_dict(dicts):
    ret = {}
    for key in dicts[0].keys():
        if len(dicts[0][key].shape) == 0:
            continue
        ret[key] = torch.cat([dic[key] for dic in dicts])
    return ret
                
                
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
            
def sample_categorical_action(action_prob, candidate_ids, slate_size, with_replacement = True, 
                              batch_wise = False, return_idx = False):
    '''
    @input:
    - action_prob: (B, L)
    - candidate_ids: (B, L) or (1, L)
    - slate_size: K
    - with_replacement: sample with replacement
    - batch_wise: do batch wise candidate selection 
    '''
    if with_replacement:
        indices = Categorical(action_prob).sample(sample_shape = (slate_size,))
        indices = torch.transpose(indices, 0, 1)
    else:
        indices = torch.cat([torch.multinomial(prob, slate_size, replacement = False).view(1,-1) \
                             for prob in action_prob], dim = 0)
    action = torch.gather(candidate_ids,1,indices) if batch_wise else candidate_ids[indices]
    if return_idx:
        return action.detach(), indices.detach()
    else:
        return action.detach()


            
#######################################
#              Learning               #
#######################################

class LinearScheduler(object):
    '''
    Code used in DQN: https://github.com/dxyang/DQN_pytorch/blob/master/utils/schedules.py
    '''
    
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p            = final_p
        self.initial_p          = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction  = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)
    
    
class SinScheduler(object):
    '''
    Code used in DQN: https://github.com/dxyang/DQN_pytorch/blob/master/utils/schedules.py
    '''
    
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p            = final_p
        self.initial_p          = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction  = np.sin(min(float(t) / self.schedule_timesteps, 1.0) * np.pi * 0.5)
        return self.initial_p + fraction * (self.final_p - self.initial_p)
    
def multi_hot_encode(indices, num_classes):
    # 创建一个全零张量
    multi_hot = torch.zeros(num_classes)
    # 将索引位置设置为1
    multi_hot[indices] = 1
    return multi_hot