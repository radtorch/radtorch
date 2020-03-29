import torch, torchvision, datetime, time, pickle, pydicom, os, itertools
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import metrics
from tqdm import tqdm_notebook as tqdm
from tqdm.notebook import tqdm
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from collections import Counter
from IPython.display import display

from .modelsutils import *
from .datautils import *
from .visutils import *
from .generalutils import *




def load_pipeline(target_path):
    '''
    .. include:: ./documentation/docs/pipeline.md##load_pipeline
    '''

    infile = open(target_path,'rb')
    pipeline = pickle.load(infile)
    infile.close()

    return pipeline
