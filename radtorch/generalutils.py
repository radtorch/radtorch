"""
Functions and Classes for General Purpose
"""
import torch, torchvision, datetime, time, pickle, pydicom, os, math, random, itertools, ntpath, copy
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


from bokeh.io import output_notebook, show
from math import pi
from bokeh.models import BasicTicker, ColorBar, LinearColorMapper, PrintfTickFormatter, Tabs, Panel, ColumnDataSource, Legend
from bokeh.plotting import figure, show
from bokeh.sampledata.unemployment1948 import data
from bokeh.layouts import row, gridplot, column
from bokeh.transform import factor_cmap, cumsum
from bokeh.palettes import viridis, Paired, inferno, brewer, d3, Turbo256


def getDuplicatesWithCount(listOfElems):
    """
    .. image:: pass.jpg
    """

    dictOfElems = dict()
    for elem in listOfElems:
        if elem in dictOfElems:
            dictOfElems[elem] += 1
        else:
            dictOfElems[elem] = 1
    dictOfElems = { key:value for key, value in dictOfElems.items() if value > 1}
    return dictOfElems



def export(item, path):
    outfile = open(path,'wb')
    pickle.dump(item,outfile)
    outfile.close()


# def import(path):
#     infile = open(path,'rb')
#     item = pickle.load(infile)
#     infile.close()
#     return item



def quality_check(): #WORK IN PROGRESS
  # Data
  p_list = datautils.list_of_files('/content/chest_xray/train/PNEUMONIA/')
  b_list = [i for i in p_list if 'bacteria' in i]
  v_list = [i for i in p_list if 'virus' in i]
  n_list = datautils.list_of_files('/content/chest_xray/train/NORMAL/')
  subset = 200
  data_sub = b_list[:subset]+v_list[:subset]+n_list[:subset]
  label_sub = ['virus']*subset+['normal']*subset
  data_label_df = pd.DataFrame(list(zip(data_sub, label_sub)), columns=['IMAGE_PATH', 'IMAGE_LABEL'])
  data_label_df = data_label_df.sample(frac=1).reset_index(drop=True)
  data_root = '/content/chest_xray/train/'
  success = []

  # Image Classification Pipeline QC
  print ('Starting Image Classification Pipeline Quality Check')
  ## label_from_table=False
  try:
    c = pipeline.Image_Classification(data_directory=data_root, label_from_table=False, is_dicom=False)
    print ('[PASS] label_from_table=False')
    success.append(1)
  except:
    print ('[FAIL] label_from_table=False')
    success.append(0)
    pass
  ## label_from_table=True, is_csv=False
  try:
    c = pipeline.Image_Classification(data_directory=data_root, label_from_table=True, is_csv=False, table_source=data_label_df, is_dicom=False)
    print ('[PASS] label_from_table=True, is_csv=False')
    success.append(1)
  except:
    print ('[FAIL] label_from_table=True, is_csv=False')
    success.append(0)
    pass
  ## label_from_table=True, is_csv=True
  ## is_csv=False
  ## is_dicom=True
  ## is_dicom=False
  ## normalize='auto'
  try:
    c = pipeline.Image_Classification(data_directory=data_root, normalize='auto', label_from_table=True, is_csv=False, table_source=data_label_df, is_dicom=False)
    print ('[PASS] normalize=auto')
    success.append(1)
  except:
    print ('[FAIL] normalize=auto')
    success.append(0)
    pass
  ## normalize='default'
  try:
    c = pipeline.Image_Classification(data_directory=data_root, normalize='default', label_from_table=True, is_csv=False, table_source=data_label_df, is_dicom=False)
    print ('[PASS] normalize=default')
    success.append(1)
  except:
    print ('[FAIL] normalize=default')
    success.append(0)
    pass
  ## normalize=False
  try:
    c = pipeline.Image_Classification(data_directory=data_root, normalize=False, label_from_table=True, is_csv=False, table_source=data_label_df, is_dicom=False)
    print ('[PASS] normalize=False')
    success.append(1)
  except:
    print ('[FAIL] normalize=False')
    success.append(0)
    pass
  ## balance_class=True
  try:
    c = pipeline.Image_Classification(data_directory=data_root, balance_class=True, label_from_table=False, is_dicom=False)
    print ('[PASS] balance_class=True')
    success.append(1)
  except:
    print ('[FAIL] balance_class=True')
    success.append(0)
    pass
  ## balance-class=False
  try:
    c = pipeline.Image_Classification(data_directory=data_root, balance_class=False, label_from_table=False, is_dicom=False)
    print ('[PASS] balance_class=False')
    success.append(1)
  except:
    print ('[FAIL] balance_class=False')
    success.append(0)
    pass
  ## batch_size = 16
  try:
    c = pipeline.Image_Classification(data_directory=data_root, is_dicom=False, batch_size=16)
    print ('[PASS] batch_size = 16')
    success.append(1)
  except:
    print ('[FAIL] batch_size = 16')
    success.append(0)
    pass
  ## batch_size = 8
  try:
    c = pipeline.Image_Classification(data_directory=data_root, is_dicom=False, batch_size=8)
    print ('[PASS] batch_size = 8')
    success.append(1)
  except:
    print ('[FAIL] batch_size = 8')
    success.append(0)
    pass
  ## learning_rate = 0.01
  try:
    c = pipeline.Image_Classification(data_directory=data_root, is_dicom=False, learning_rate=0.01)
    print ('[PASS] learning_rate = 0.01')
    success.append(1)
  except:
    print ('[FAIL] learning_rate = 0.01')
    success.append(0)
    pass
  ## learning_rate = 0.0001
  try:
    c = pipeline.Image_Classification(data_directory=data_root, is_dicom=False, learning_rate=0.0001)
    print ('[PASS] learning_rate = 0.0001')
    success.append(1)
  except:
    print ('[FAIL] learning_rate = 0.0001')
    success.append(0)
    pass
  ## valid_percent=0.3, test_percent=0.4
  try:
    c = pipeline.Image_Classification(data_directory=data_root, is_dicom=False, valid_percent=0.3, test_percent=0.4)
    print ('[PASS] valid_percent=0.3, test_percent=0.4')
    success.append(1)
  except:
    print ('[FAIL] valid_percent=0.3, test_percent=0.4')
    success.append(0)
    pass
  # Models _ trained
  for i in ['vgg16', 'vgg19', 'resnet50', 'resnet152']:
    try:
      c = pipeline.Image_Classification(data_directory=data_root, is_dicom=False, model_arch=i)
      print ('[PASS] model=', i, 'trained=True')
      success.append(1)
    except:
      print ('[FAIL] model=', i, 'trained=True')
      success.append(0)
      pass
  # Models _ trained = False
  for i in ['vgg16', 'vgg19', 'resnet50', 'resnet152']:
    try:
      c = pipeline.Image_Classification(data_directory=data_root, is_dicom=False, model_arch=i, pre_trained=False)
      print ('[PASS] model=', i, 'trained=False')
      success.append(1)
    except:
      print ('[FAIL] model=', i, 'trained=False')
      success.append(0)
      pass

  c = pipeline.Image_Classification(data_directory=data_root, train_epochs=2, label_from_table=True, is_csv=False, table_source=data_label_df, is_dicom=False)
  try:
    c.info()
    print ('[PASS] classification pipeline info')
    success.append(1)
  except:
    print ('[FAIL] classification pipeline info')
    success.append(0)
    pass
  try:
    c.dataset_info()
    print ('[PASS] classification pipeline dataset_info')
    success.append(1)
  except:
    print ('[FAIL] classification pipeline dataset_info')
    success.append(0)
    pass
  try:
    c.sample()
    print ('[PASS] classification pipeline sample')
    success.append(1)
  except:
    print ('[FAIL] classification pipeline sample')
    success.append(0)
    pass
  try:
    c.run()
    print ('[PASS] classification pipeline train')
    success.append(1)
  except:
    print ('[FAIL] classification pipeline train')
    success.append(0)
    pass
  try:
    c.metrics()
    print ('[PASS] classification pipeline metrics')
    success.append(1)
  except:
    print ('[FAIL] classification pipeline metrics')
    success.append(0)
    pass
  try:
    c.confusion_matrix()
    print ('[PASS] classification pipeline confusion_matrix')
    success.append(1)
  except:
    print ('[FAIL] classification pipeline confusion_matrix')
    success.append(0)
    pass
  try:
    c.misclassified()
    print ('[PASS] classification pipeline misclassified')
    success.append(1)
  except:
    print ('[FAIL] classification pipeline misclassified')
    success.append(0)
    pass
  try:
    c.roc()
    print ('[PASS] classification pipeline roc')
    success.append(1)
  except:
    print ('[FAIL] classification pipeline roc')
    success.append(0)
    pass


  # Feature Extraction Pipeline QC
  # Compare Image Classification Pipeline QC

  print ('Release passed', sum(success), '/', len(success), 'tests.')
