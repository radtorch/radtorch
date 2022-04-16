import torch, random
import numpy as np
from datetime import datetime



def set_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def path_fix(root):
    if root.endswith("/"):
        return root
    return (root + "/" )


def current_time(human=True):
    if human:
        dt_string = (datetime.now()).strftime("%d-%m-%Y %H:%M:%S")
    else:
        dt_string = (datetime.now()).strftime("%d%m%Y%H%M%S")
    return dt_string


def message(msg, msg_type=''):
    print ('['+current_time()+']', msg_type, msg)


def select_device(device='auto'):
    if device=='auto':
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device)
