"""
Functions and Classes for Data Handling and PreProcessing
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


from radtorch.dicomutils import  *
from radtorch.visutils import *
from radtorch.settings import *



def over_sample(shuffle=True, **kwargs):
    '''
    Arguments:
    ----------
    dataset: target dataset.
    shuffle: True to shuffle.
    '''
    balanced_dataset = copy.deepcopy(kwargs['dataset'])
    max_size = balanced_dataset.input_data[balanced_dataset.image_label_col].value_counts().max()
    lst = [balanced_dataset.input_data]
    for class_index, group in balanced_dataset.input_data.groupby(balanced_dataset.image_label_col):
      lst.append(group.sample(max_size-len(group), replace=True))
    balanced_dataframe = pd.concat(lst)
    if shuffle:
        balanced_dataframe = balanced_dataframe.sample(frac=1).reset_index(drop=True)
    balanced_dataset.input_data = balanced_dataframe
    return balanced_dataset


def calculate_mean_std(dataloader):
    '''
    Source
    -------
    https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/6
    '''
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, labels, paths in dataloader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
    mean /= nb_samples
    std /= nb_samples
    return (mean, std)


def split_dataset(dataset, valid_percent=0.2, test_percent=0.2, equal_class_split=True, shuffle=True):
    num_all = len(dataset)
    train_percent = 1.0 - (valid_percent+test_percent)

    num_classes = dataset.input_data[dataset.image_label_col].unique()

    classes_df = []
    for i in num_classes:
        temp_df = dataset.input_data.loc[dataset.input_data[dataset.image_label_col]==i]
        if shuffle:
          temp_df = temp_df.sample(frac=1).reset_index(drop=True)
        train, validate, test = np.split(temp_df.sample(frac=1), [int(train_percent*len(temp_df)), int((train_percent+valid_percent)*len(temp_df))])
        classes_df.append((train, validate, test))

    if test_percent != 0:
        train_df = (pd.concat([i[0] for i in classes_df])).sample(frac=1).reset_index(drop=True)
        valid_df = (pd.concat([i[1] for i in classes_df])).sample(frac=1).reset_index(drop=True)
        test_df = (pd.concat([i[2] for i in classes_df])).sample(frac=1).reset_index(drop=True)

        train_ds = dataset_from_table(data_directory=dataset.data_directory,is_dicom=dataset.is_dicom, is_csv=False, input_source=train_df, mode=dataset.mode, wl=dataset.wl, trans=dataset.trans)
        valid_ds = dataset_from_table(data_directory=dataset.data_directory,is_dicom=dataset.is_dicom, is_csv=False, input_source=valid_df, mode=dataset.mode, wl=dataset.wl, trans=dataset.trans)
        test_ds = dataset_from_table(data_directory=dataset.data_directory,is_dicom=dataset.is_dicom, is_csv=False, input_source=test_df, mode=dataset.mode, wl=dataset.wl, trans=dataset.trans)

        return  train_ds, valid_ds, test_ds
    else:
        train_df = (pd.concat([i[0] for i in classes_df])).sample(frac=1).reset_index(drop=True)
        valid_df = (pd.concat([i[1] for i in classes_df])).sample(frac=1).reset_index(drop=True)

        train_ds = dataset_from_table(data_directory=dataset.data_directory,is_dicom=dataset.is_dicom, is_csv=False, input_source=train_df, mode=dataset.mode, wl=dataset.wl, trans=dataset.trans)
        valid_ds = dataset_from_table(data_directory=dataset.data_directory,is_dicom=dataset.is_dicom, is_csv=False, input_source=valid_df, mode=dataset.mode, wl=dataset.wl, trans=dataset.trans)

        return  train_ds, valid_ds


def set_random_seed(seed):
    """
    .. include:: ./documentation/docs/datautils.md##set_random_seed
    """
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        print ('Random Seed Set Successfully')
    except:
        raise TypeError('Error. Could not set Random Seed. Please check again.')
        pass


def list_of_files(root):
    """
    .. include:: ./documentation/docs/datautils.md##list_of_files
    """

    listOfFile = os.listdir(root)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(root, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + list_of_files(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles


def path_to_class(filepath):
    """
    .. include:: ./documentation/docs/datautils.md##path_to_class
    """

    item_class = (Path(filepath)).parts
    return item_class[-2]


def root_to_class(root):

    """
    .. include:: ./documentation/docs/datautils.md##root_to_class
    """

    classes = [d.name for d in os.scandir(root) if d.is_dir()]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def class_to_idx(classes):
    """
    .. include:: ./documentation/docs/datautils.md##class_to_idx
    """

    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return class_to_idx


class dataset_from_table(Dataset):
    """
    .. include:: ./documentation/docs/datautils.md##dataset_from_table
    """

    def __init__(self,
                data_directory,
                is_csv=True,
                is_dicom=True,
                input_source=None,
                img_path_column='IMAGE_PATH',
                img_label_column='IMAGE_LABEL',
                multi_label = False,
                mode='RAW',
                wl=None,
                trans=transforms.Compose([transforms.ToTensor()])):

        self.data_directory = data_directory
        self.is_csv = is_csv
        self.is_dicom = is_dicom
        self.input_source = input_source
        self.image_path_col = img_path_column
        self.image_label_col = img_label_column
        self.mode = mode
        self.wl = wl
        self.trans = trans
        self.multi_label = multi_label


        if self.is_csv:
            if self.input_source == None:
                print ('Error! No source for csv data was selected. Please use csv_file argument to supply a csv file')
            else:
                self.input_data = pd.read_csv(self.input_source)
        else:
            self.input_data = self.input_source


        if self.is_dicom:
            self.dataset_files = [x for x in (self.input_data[self.image_path_col].tolist()) if x[-3:] == 'dcm'] # Returns only DICOM files from folder
        else:
            self.dataset_files = [x for x in (self.input_data[self.image_path_col].tolist()) if x.lower().endswith(IMG_EXTENSIONS)]


        if self.multi_label == True:
            self.classes = list(np.unique([item for t in self.input_data[self.image_label_col].to_numpy() for item in t]))
            self.class_to_idx = class_to_idx(self.classes)
            self.multi_label_idx = []
            for i, row in self.input_data.iterrows():
                t = []
                for u in self.classes:
                    if u in row[self.image_label_col]:
                        t.append(1)
                    else:
                        t.append(0)
                self.multi_label_idx.append(t)
            self.input_data['MULTI_LABEL_IDX'] = self.multi_label_idx

        else:
            self.classes = np.unique(list(self.input_data[self.image_label_col]))
            self.class_to_idx = class_to_idx(self.classes)



        if len(self.dataset_files)==0:
            print ('Error! No data files found in directory:', self.data_directory)

        if len(self.classes)    ==0:
            print ('Error! No classes extracted from directory:', self.data_directory)



    def __getitem__(self, index):
        image_path = self.input_data.iloc[index][self.image_path_col]
        if self.is_dicom:
            image = dicom_to_narray(image_path, self.mode, self.wl)
            image = Image.fromarray(image)

        else:
            image = Image.open(image_path).convert('RGB')

        image = self.trans(image)

        if self.multi_label == True:
            label = self.input_data.iloc[index][self.image_label_col]
            label_idx = self.input_data.iloc[index]['MULTI_LABEL_IDX']

        else:
            label = self.input_data.iloc[index][self.image_label_col]
            label_idx = [v for k, v in self.class_to_idx.items() if k == label][0]


        return image, label_idx, image_path

    def __len__(self):
        return len(self.dataset_files)

    def classes(self):
        return self.classes

    def class_to_idx(self):
        return self.class_to_idx

    def info(self):
        return show_dataset_info(self)


class dataset_from_folder(Dataset):
    """
    .. include:: ./documentation/docs/datautils.md##dataset_from_folder
    """

    def __init__(self,
                data_directory,
                is_dicom=True,
                mode='RAW',
                wl=None,
                trans=transforms.Compose([transforms.ToTensor()])):

        self.data_directory = data_directory
        self.is_dicom = is_dicom
        self.mode = mode
        self.wl = wl
        self.trans = trans
        self.classes, self.class_to_idx = root_to_class(self.data_directory)
        self.all_files = list_of_files(self.data_directory)
        self.all_classes = [path_to_class(i) for i in self.all_files]
        self.image_path_col = 'IMAGE_PATH'
        self.image_label_col = 'IMAGE_LABEL'
        self.input_data = pd.DataFrame(list(zip(self.all_files, self.all_classes)), columns=[self.image_path_col, self.image_label_col])


        if self.is_dicom:
            self.dataset_files = [x for x in self.all_files if x[-3:] == 'dcm'] # Returns only DICOM files from folder
        else:
            self.dataset_files = [x for x in self.all_files]


        if len(self.dataset_files)==0:
            print ('Error! No data files found in directory:', self.data_directory)

        if len(self.classes)==0:
            print ('Error! No classes extracted from directory:', self.data_directory)

    def __getitem__(self, index):
        image_path = self.dataset_files[index]
        if self.is_dicom:
            image = dicom_to_narray(image_path, self.mode, self.wl)
            image = Image.fromarray(image)

        else:
            image = Image.open(image_path).convert('RGB')

        image = self.trans(image)

        label = path_to_class(image_path)
        label_idx = [v for k, v in self.class_to_idx.items() if k == label][0]

        return image, label_idx, image_path

    def __len__(self):
        return len(self.dataset_files)

    def classes(self):
        return self.classes

    def class_to_idx(self):
        return self.class_to_idx

    def info(self):
        return show_dataset_info(self)


def load_predefined_datatables(*args, **kwargs):
    train_data_set = dataset_from_table(
                                        data_directory=kwargs['data_directory'],
                                        is_csv=kwargs['is_csv'],
                                        is_dicom=kwargs['is_dicom'],
                                        input_source=kwargs['predefined_datasets']['train'],
                                        img_path_column=kwargs['path_col'],
                                        img_label_column=kwargs['label_col'],
                                        mode=kwargs['mode'],
                                        wl=kwargs['wl'],
                                        trans=kwargs['transformations'])

    valid_data_set = dataset_from_table(
                                        data_directory=kwargs['data_directory'],
                                        is_csv=kwargs['is_csv'],
                                        is_dicom=kwargs['is_dicom'],
                                        input_source=kwargs['predefined_datasets']['valid'],
                                        img_path_column=kwargs['path_col'],
                                        img_label_column=kwargs['label_col'],
                                        mode=kwargs['mode'],
                                        wl=kwargs['wl'],
                                        trans=kwargs['transformations'])

    test_data_set = dataset_from_table(
                                        data_directory=kwargs['data_directory'],
                                        is_csv=kwargs['is_csv'],
                                        is_dicom=kwargs['is_dicom'],
                                        input_source=kwargs['predefined_datasets']['test'],
                                        img_path_column=kwargs['path_col'],
                                        img_label_column=kwargs['label_col'],
                                        mode=kwargs['mode'],
                                        wl=kwargs['wl'],
                                        trans=kwargs['transformations'])

    return train_data_set, valid_data_set, test_data_set





##
