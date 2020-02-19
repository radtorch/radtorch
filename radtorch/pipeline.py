

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


from radtorch.modelsutils import create_model, create_loss_function, train_model
from radtorch.datautils import dataset_from_folder, dataset_from_table
from radtorch.visutils import show_dataset_info, show_dataloader_sample, show_metrics


class Image_Classification():
    """
    Creates an Image Classification Pipeline.

    Inputs:
        data_directory: **[REQUIRED]** [str] target data directory.
        is_dicom: [boolean] True for DICOM images, False for regular images.(default=True)
        label_from_table:[boolean] True if labels are to extracted from table, False if labels are to be extracted from subfolders. (default=False)
        is_csv: [boolean] True for csv, False for pandas dataframe.
        table_source: [str or pandas dataframe object] source for labelling data. (default=None)
                      This is path to csv file or name of pandas dataframe if pandas to be used.
        mode: [str] output mode for DICOM images only. (default='RAW')
                    options:
                        RAW= Raw pixels,
                        HU= Image converted to Hounsefield Units,
                        WIN= 'window' image windowed to certain W and L,
                        MWIN = 'multi-window' converts image to 3 windowed images of different W and L (specified in wl argument) stacked together].
        wl: [list] list of lists of combinations of window level and widths to be used with WIN and MWIN. (default=None)
                    In the form of : [[Level,Width], [Level,Width],...].
                    Only 3 combinations are allowed for MWIN (for now).
        trans:[pytorch transforms] pytroch transforms to be performed on the dataset. (default=Convert to tensor)
        batch_size: [int] batch size of the dataset (default=16)
        test_split: [float] percentage of dataset to use for validation. Float value between 0 and 1.0. (default=0.2)
        model_arch: [str] PyTorch neural network architecture (default='vgg16')
        pre_trained: [boolen] Load the pretrained weights of the neural network.(default=True)
        num_input_channels: [int] Number of input image channels. Grayscale DICOM images usually have 1 channel. Colored images have 3. (default=1)
        train_epochs: [int] Number of training epochs. (default=20)
        learning_rate: [float] training learning rate. (default = 0.001)
        loss_function: [str] training loss function. (default='NLLLoss')
        optimizer: [str] Optimizer to be used during training. (default='Adam')
        device: [str] device to be used for training (default='cpu'). This can be 'cpu' or 'gpu'. (default='cpu')

    Outputs:
        Output: Image Classification Model

    Examples:
    ```
    ```

    .. image:: pass.jpg
    """

    def __init__(
    self,
    data_directory,
    optimizer='Adam',
    trans=transforms.Compose([transforms.ToTensor()]),
    is_dicom=True,
    label_from_table=False,
    is_csv=None,
    table_source=None,
    mode='RAW',
    wl=None,
    batch_size=16,
    test_split = 0.2,
    model_arch='vgg16',
    pre_trained=True,
    num_input_channels=1,
    train_epochs=20,
    learning_rate=0.001,
    loss_function='NLLLoss',
    device='cpu'):
        self.data_directory = data_directory
        self.label_from_table = label_from_table
        self.is_csv = is_csv
        self.is_dicom = is_dicom
        self.table_source = table_source
        self.mode = mode
        self.wl = wl
        self.trans = trans
        self.batch_size = batch_size
        self.test_split = test_split
        self.model_arch = model_arch
        self.pre_trained = pre_trained
        self.num_input_channels = num_input_channels
        self.train_epochs = train_epochs
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = device
        self.path_col = 'IMAGE_PATH'
        self.label_col = 'IMAGE_LABEL'

        # Create DataSet
        if self.label_from_table == True:
            self.data_set = dataset_from_table(
                    data_directory=self.data_directory,
                    is_csv=self.is_csv,
                    is_dicom=self.is_dicom,
                    input_source=self.table_source,
                    img_path_column=self.path_col,
                    img_label_column=self.label_col,
                    mode=self.mode,
                    wl=self.wl,
                    trans=self.trans)

        else:
            self.data_set = dataset_from_folder(
                        data_directory=self.data_directory,
                        is_dicom=self.is_dicom,
                        mode=self.mode,
                        wl=self.wl,
                        trans=self.trans)

        # Create DataLoader

        valid_size = int(self.test_split*len(self.data_set))

        self.valid_data_set, self.train_data_set = torch.utils.data.random_split(self.data_set, [valid_size,len(self.data_set)-valid_size])

        self.train_data_loader = torch.utils.data.DataLoader(
                                                    self.train_data_set,
                                                    batch_size=self.batch_size,
                                                    shuffle=True)

        self.valid_data_loader = torch.utils.data.DataLoader(
                                                    self.valid_data_set,
                                                    batch_size=self.batch_size,
                                                    shuffle=True)


        self.num_output_classes = len(self.data_set.classes)


        # Create Model
        self.train_model = create_model(
                                    model_arch=self.model_arch,
                                    input_channels=self.num_input_channels,
                                    output_classes=self.num_output_classes,
                                    pre_trained=self.pre_trained)


        self.loss_function = create_loss_function(self.loss_function)


        if self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.train_model.parameters(), lr=self.learning_rate)



    def show_info(self):
        '''
        Displays Image Classification Pipeline Parameters.
        '''
        print ('RADTorch Image Classification Pipeline Parameters')
        for key, value in self.__dict__.items():
            if key != 'trans':
                print('>', key,'=',value)
        print ('Train Dataset Size =', len(self.train_data_set))
        print ('Valid Dataset Size =', len(self.valid_data_set))

    def show_dataset_info(self,):
        '''
        Displays Dataset Information.
        '''
        print (show_dataset_info(self.data_set))
        print ('Train Dataset Size ', len(self.train_data_set))
        print ('Valid Dataset Size ', len(self.valid_data_set))

    def show_sample(self, num_of_images_per_row=5, fig_size=(10,10), show_labels=True):
        '''
        Displays sample of the training dataset.
        '''
        return show_dataloader_sample(dataloader=self.train_data_loader, num_of_images_per_row=num_of_images_per_row, figsize=fig_size, show_labels=show_labels)

    def train_classifier(self):
        '''
        Train the created image classifier.
        '''
        self.trained_model, self.train_metrics = train_model(
                                                model = self.train_model,
                                                train_data_loader = self.train_data_loader,
                                                valid_data_loader = self.valid_data_loader,
                                                train_data_set = self.train_data_set,
                                                valid_data_set = self.valid_data_set,
                                                loss_criterion = self.loss_function,
                                                optimizer = self.optimizer,
                                                epochs = self.train_epochs,
                                                device = self.device)

    def show_train_metrics(self):
        '''
        Display the training metrics
        '''
        show_metrics(self.train_metrics)


    def export_classifier(self,output_path):
        torch.save(self.trained_model, output_path)
        print ('Trained classifier exported successfully.')



#
# class Pipeline():
#     def __init__(
#     self,
#     data_directory,
#     label_from_table,
#     is_csv,
#     is_dicom,
#     table_source,
#     mode,
#     wl,
#     trans,
#     batch_size,
#     model_arch,
#     pre_trained,
#     num_input_channels,
#     train_epochs,
#     learning_rate,
#     loss_function,
#     optimizer,):
#         self.data_directory = data_directory
#         self.label_from_table = label_from_table
#         self.is_csv = is_csv
#         self.is_dicom = is_dicom
#         elf.table_source = table_source
#         self.mode = mode
#         self.wl = wl
#         self.trans = trans
#         self.batch_size = batch_size
#         self.model_arch = model_arch
#         self.pre_trained = pre_trained
#         self.num_input_channels = num_input_channels
#         self.train_epochs = train_epochs
#         self.learning_rate = learning_rate
#         self.loss_function = loss_function
#         self.optimizer = optimizer
#         self.path_col = 'IMAGE_PATH'
#         self.label_col = 'IMAGE_LABEL'
#
#         # Create DataSet
#         if self.label_from_table == True:
#             self.data_set = dataset_from_table(
#                     data_directory=self.data_directory,
#                     is_csv=self.is_csv,
#                     is_dicom=self.is_dicom,
#                     input_source=self.table_source,
#                     img_path_column=self.path_col,
#                     img_label_column=self.label_col,
#                     mode=self.mode,
#                     wl=self.wl,
#                     trans=self.trans)
#
#         else:
#             self.data_set = dataset_from_folder(
#                         data_directory=self.data_directory,
#                         is_dicom=self.is_dicom,
#                         mode=self.mode,
#                         wl=self.wl,
#                         trans=self.trans)
#
#         # Create DataLoader
#         self.data_loader = torch.utils.data.DataLoader(
#                                                     self.data_set,
#                                                     batch_size=self.batch_size,
#                                                     shuffle=True)
#
#
#         self.num_output_classes = len(self.data_set.classes)
#
#
#         # Create Model
#         self.train_model = create_model(
#                                     model_arch=self.model_arch,
#                                     input_channels=elf.num_input_channels,
#                                     output_classes=self.num_output_classes,
#                                     pre_trained=self.pre_trained)
#
#
#         self.loss_function = create_loss_function(self.loss_function)
#
#
#         if self.optimizer == 'Adam':
#             self.optimizer = torch.nn.Adam(self.train_model.parameters(), lr=self.learning_rate)
#
#
#
#
#     def info(self):
#         print ('''RADTorch Pipeline Attributes
#         ''')
#         for key, value in self.__dict__.items():
#             if key != 'trans':
#                 print('>', key,'=',value)
#
#     def dataset_info(self,):
#         return show_dataset_info(self.data_set)
#
#     def sample(self, num_of_images_per_row=5, fig_size=(10,10), show_labels=True):
#         return show_dataloader_sample(dataloader=self.data_loader, num_of_images_per_row=num_of_images_per_row, figsize=fig_size, show_labels=show_labels)
#
#
# ##
## Components of Pipeline

##Data
# data directory
# is csv
# is dicom
# csv/table location
# columns
# mode
# wl
# transforms
# batch size


## Model
# input channels
# output classes

## Training
# epochs
# learning rate
# optimizer
# loss function


#### Pipeline functions
# info
# dataset_info
# sample
# set_dataset
# set_model
# set_learning_rate
# set_transforms