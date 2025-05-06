import os
import random
import logging
import pickle
import numpy as np
import torch


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def print_and_log(message):
    # logging.info(message)
    print(message) 
    
    
def if_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)