import torch, torchvision, datetime, time, pickle, pydicom, os
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
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image



def path_to_class(filepath):
    '''
    Creates a class from the folder name of a file
    Inputs:
        filepath: [str] path to target file
    Output:
        [str] folder name / class name
    '''
    item_class = (Path(filepath)).parts
    return item_class[-2]

def root_to_class(root):

    """
    Creates list of classes and dictionary of classes and idx in a given data root.
    All first level subfolders within the root are converted into classes and given class id.
    Inputs:
        root: [str] path of the desired root.
    Outputs:
        [tuple] classes: [list] of generated classes, class_to_idx: [dictionary] of classes and class id numbers
    """
    classes = [d.name for d in os.scandir(root) if d.is_dir()]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def class_to_idx(classes):
    """
    Creates a dictionary of classes to classes idx from provided list of classes
    Inputs:
        classes: [list] list of target classes
    Output:
        [dictionary] of classes and class id numbers
    """
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return class_to_idx

class dataset_from_table(Dataset):
    """
    Creates a dataset from directory with labels from table which can be either a excel sheet or pandas dataframe.
    Default values for columns are: "IMAGE_PATH", "IMAGE_LABEL". These can be changed as needed.
    Inputs:
        data_directory: [str] target data directory.
        is_csv: [boolean] True for csv, False for pandas dataframe. (default=True)
        is_dicom: [boolean] True for DICOM images, False for regular images.(default=True)
        input_source: [str or pandas dataframe object] source for labelling data.
                      This is path to csv file or name of pandas dataframe if pandas to be used.
        img_path_column = [list] name of the image path column in data input
        img_label_column = [str] name of label column in the data input
        mode: [str] output mode for DICOM images only.
                    options: RAW= Raw pixels,
                    HU= Image converted to Hounsefield Units,
                    WIN= 'window' image windowed to certain W and L,
                    MWIN = 'multi-window' converts image to 3 windowed images of different W and L (specified in wl argument) stacked together].
        wl: [list] list of lists of combinations of window level and widths to be used with WIN and MWIN. (default=None)
                    In the form of : [[Level,Width], [Level,Width],...].
                    Only 3 combinations are allowed for MWIN (for now).
        transforms: [pytorch transforms] pytroch transforms to be performed on the dataset.
    """
    def __init__(self,
                data_directory,
                is_csv=True,
                is_dicom=True,
                input_source=None,
                img_path_column='IMAGE_PATH',
                img_label_column='IMAGE_LABEL',
                mode='RAW',
                wl=None, trans=None):

        self.data_directory = data_directory
        self.is_csv = is_csv
        self.is_dicom = is_dicom
        self.input_source = input_source
        self.image_path_col = img_path_column
        self.image_label_col = img_label_column
        self.mode = mode
        self.wl = wl
        self.trans = trans


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
            self.dataset_files = [x for x in (self.input_data[self.image_path_col].tolist())]
        # self.classes = self.input_data.self.image_label_col.unique()
        self.classes = np.unique(list(self.input_data[self.image_label_col]))
        self.class_to_idx = class_to_idx(self.classes)

        if len(self.dataset_files)==0:
            print ('Error! No data files found in directory:', self.data_directory)

        if len(self.classes)==0:
            print ('Error! No classes extracted from directory:', self.data_directory)

    def __getitem__(self, index):
        image_path = self.input_data.iloc[index][self.image_path_col]
        if self.is_dicom:
            image = convert_dcm_to_np(image_path, self.mode, self.wl)

        else:
            image = Image.open(img_path).convert('RGB')

        if self.trans:
            image = self.trans(image)

        label = self.input_data.iloc[index][self.image_label_col]
        label_idx = [v for k, v in self.class_to_idx.items() if k == label][0]

        return image, label_idx

    def __len__(self):
        return len(self.dataset_files)

    def classes(self):
        return self.classes

    def class_to_idx(self):
        return self.class_to_idx

    def info(self):
        return show_dataset_info(self)
