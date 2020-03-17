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
from pathlib import Path


from radtorch.dicomutils import  dicom_to_narray, window_dicom, dicom_to_pil
from radtorch.visutils import show_dataset_info


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def set_random_seed(seed):
    """
    .. include:: ./documentation/docs/datautils.md##set_random_seed
    """
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        print ('random seed set successfully')
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
                wl=None, trans=transforms.Compose([transforms.ToTensor()])):

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








##
