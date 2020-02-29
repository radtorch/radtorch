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



def list_of_files(root):
    """
        Create a list of file paths from a root folder and its sub directories.

          **Arguments**

          - root: _(str)_ path of target folder.

          **Output**

          - list of file paths.


          **Example**

            root_path = 'root/'
            list_of_files(root_path)

          <!-- **** -->

            ['root/folder1/0000.dcm', 'root/folder1/0001.dcm', 'root/folder2/0000.dcm', ...]

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
    Creates a class name from the immediate parent folder of a target file.

      **Arguments**

      - filepath: _(str)_ path to target file.

      **Output**

      - _(str)_ folder name / class name.


      **Example**

        file_path = 'root/folder1/folder2/0000.dcm'
        path_to_class(file_path)

      <!-- **** -->

        'folder2'
    """

    item_class = (Path(filepath)).parts
    return item_class[-2]

def root_to_class(root):

    """
    Creates list of classes and dictionary of classes and idx in a given data root. All first level subfolders within the root are converted into classes and given class id.


      **Arguments**

      - root: _(str)_ path of target root.

      **Output**

      - _(tuple)_ of
        - classes: _(list)_ of generated classes,
        - class_to_idx: _(dictionary)_ of classes and class id numbers


      **Example**

      This example assumes that root folder contains 3 folders (folder1, folder2 and folder3) each contains images of 1 class.

        root_folder = 'root/'
        root_to_class(root_folder)

      <!-- **** -->

        ['folder1', 'folder2', 'folder3'], {'folder1':0, 'folder2':1, 'folder3':2}

    """

    classes = [d.name for d in os.scandir(root) if d.is_dir()]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def class_to_idx(classes):
    """
      Creates a dictionary of classes to classes idx from provided list of classes

      **Arguments**

      - classes: _(list)_ list of classes

      **Output**

      - Output: _(dictionary)_ dictionary of classes to class idx


      **Example**

        class_list = ['class1','class4', 'class2', 'class3']
        class_to_idx(class_list)

      <!-- **** -->

        {'class1':0, 'class2':1, 'class3':2, 'class4':3}
    """

    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return class_to_idx

class dataset_from_table(Dataset):
    """
    Creates a dataset from a root directory using subdirectories as classes/labels.

    **Parameters**

    - data_director: _(str)_ target data root directory.

    - is_dicom: _(boolean)_ True for DICOM images, False for regular images.(default=True)

    - mode: _(str)_ output mode for DICOM images only. options: RAW= Raw pixels, HU= Image converted to Hounsefield Units, WIN= 'window' image windowed to certain W and L, MWIN = 'multi-window' converts image to 3 windowed images of different W and L (specified in wl argument) stacked together].

    - wl: _(list)_ list of lists of combinations of window level and widths to be used with WIN and MWIN. In the form of : [[Level,Width], [Level,Width],…]. Only 3 combinations are allowed for MWIN (for now). (default=None)

    - trans: _(pytorch transforms)_ pytroch transforms to be performed on the dataset. (default=Convert to tensor)


    **Methods**

    - **class_to_idx**

          Returns dictionary of dataset classes and corresponding class id.

    - **classes**

          Returns list of dataset classes

    - **info**

          Returns detailed information of the dataset.

    """

    def __init__(self,
                data_directory,
                is_csv=True,
                is_dicom=True,
                input_source=None,
                img_path_column='IMAGE_PATH',
                img_label_column='IMAGE_LABEL',
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
            image = dicom_to_narray(image_path, self.mode, self.wl)
            image = Image.fromarray(image)

        else:
            image = Image.open(image_path).convert('RGB')

        image = self.trans(image)

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
    Creates a dataset using labels and filepaths from a table which can be either a excel sheet or pandas dataframe.


    **Parameters**

    - data_directory: _(str)_ target data directory.

    - is_csv: _(boolean)_ True for csv, False for pandas dataframe. (default=True)

    - is_dicom: _(boolean)_ True for DICOM images, False for regular images.(default=True)

    - input_source: _(str or pandas dataframe object)_ source for labelling data. This is path to csv file or name of pandas dataframe if pandas to be used.

    - img_path_column: _(str)_  name of the image path column in data input. (default = "IMAGE_PATH")

    - img_label_column: _(str)_  name of label column in the data input (default = "IMAGE_LABEL")

    - mode: _(str)_  output mode for DICOM images only. options: RAW= Raw pixels, HU= Image converted to Hounsefield Units, WIN= 'window' image windowed to certain W and L, MWIN = 'multi-window' converts image to 3 windowed images of different W and L (specified in wl argument) stacked together].

    - wl: _(list)_  list of lists of combinations of window level and widths to be used with WIN and MWIN.In the form of : [[Level,Width], [Level,Width],…]. Only 3 combinations are allowed for MWIN (for now).  (default=None)

    - transforms: _(pytorch transforms)_ pytroch transforms to be performed on the dataset. (default=Convert to tensor)


    **Methods**

    - **class_to_idx**

          Returns dictionary of dataset classes and corresponding class id.

    - **classes**

          Returns list of dataset classes

    - **info**

          Returns detailed information of the dataset.
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
