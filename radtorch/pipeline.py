

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


from radtorch.modelsutils import create_model, create_loss_function, train_model, model_inference, model_dict, create_optimizer
from radtorch.datautils import dataset_from_folder, dataset_from_table
from radtorch.visutils import show_dataset_info, show_dataloader_sample, show_metrics, show_confusion_matrix, show_roc, show_nn_roc




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
        transformations:[pytorch transforms] pytroch transforms to be performed on the dataset. (default=Convert to tensor)
        custom_resize: [int] by default, a radtorch pipeline will resize the input images into the default training model input image size as demosntrated in the table shown in radtorch home page. This default size can be changed here if needed.
        batch_size: [int] batch size of the dataset (default=16)
        test_split: [float] percentage of dataset to use for validation. Float value between 0 and 1.0. (default=0.2)
        model_arch: [str] PyTorch neural network architecture (default='vgg16')
        pre_trained: [boolen] Load the pretrained weights of the neural network. If False, the last layer is only retrained = Transfer Learning. (default=True)
        unfreeze_weights: [boolen] if True, all model weights, not just final layer, will be retrained. (default=False)
        train_epochs: [int] Number of training epochs. (default=20)
        learning_rate: [float] training learning rate. (default = 0.0001)
        loss_function: [str] training loss function. (default='CrossEntropyLoss')
        optimizer: [str] Optimizer to be used during training. (default='Adam')
        device: [str] device to be used for training. This can be adjusted to 'cpu' or 'cuda'. If nothing is selected, the pipeline automatically detects if cuda is available and trains on it.

    Outputs:
        Output: Image Classification Model

    Examples:
    ```
    from radtorch import pipeline

    classifier = pipeline.Image_Classification(data_directory='path to data')
    classifier.train()
    classifier.metrics()

    ```

    .. image:: pass.jpg
    """

    def __init__(
    self,
    data_directory,
    transformations='default',
    custom_resize = 'default',
    device='default',
    optimizer='Adam',
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
    unfreeze_weights=False,
    train_epochs=20,
    learning_rate=0.0001,
    loss_function='CrossEntropyLoss'):
        self.data_directory = data_directory
        self.label_from_table = label_from_table
        self.is_csv = is_csv
        self.is_dicom = is_dicom
        self.table_source = table_source
        self.mode = mode
        self.wl = wl

        if custom_resize=='default':
            self.input_resize = model_dict[model_arch]['input_size']
        else:
            self.input_resize = custom_resize

        if transformations == 'default':
            if self.is_dicom == True:
                self.transformations = transforms.Compose([
                        transforms.Resize((self.input_resize, self.input_resize)),
                        transforms.transforms.Grayscale(3),
                        transforms.ToTensor()])
            else:
                self.transformations = transforms.Compose([
                        transforms.Resize((self.input_resize, self.input_resize)),
                        transforms.ToTensor()])
        else:
            self.transformations = transformations

        self.batch_size = batch_size
        self.test_split = test_split
        self.model_arch = model_arch
        self.pre_trained = pre_trained
        self.unfreeze_weights = unfreeze_weights
        self.train_epochs = train_epochs
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.path_col = 'IMAGE_PATH'
        self.label_col = 'IMAGE_LABEL'
        if device == 'default':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device == device

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
                    trans=self.transformations)

        else:
            self.data_set = dataset_from_folder(
                        data_directory=self.data_directory,
                        is_dicom=self.is_dicom,
                        mode=self.mode,
                        wl=self.wl,
                        trans=self.transformations)



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

        self.train_model = create_model(
                                    model_arch=self.model_arch,
                                    output_classes=self.num_output_classes,
                                    pre_trained=self.pre_trained,
                                    unfreeze_weights = self.unfreeze_weights
                                    )

        self.train_model = self.train_model.to(self.device)

        self.loss_function = create_loss_function(self.loss_function)

        self.optimizer = create_optimizer(traning_model=self.train_model, optimizer_type=optimizer, learning_rate=self.learning_rate)


    def info(self):
        '''
        Displays Image Classification Pipeline Parameters.
        '''
        print ('RADTorch Image Classification Pipeline Parameters')
        for key, value in self.__dict__.items():
            if key != 'trans':
                print('>', key,'=',value)
        print ('Train Dataset Size =', len(self.train_data_set))
        print ('Valid Dataset Size =', len(self.valid_data_set))

    def dataset_info(self):
        '''
        Displays Dataset Information.
        '''
        print (show_dataset_info(self.data_set))
        print ('Train Dataset Size ', len(self.train_data_set))
        print ('Valid Dataset Size ', len(self.valid_data_set))

    def sample(self, num_of_images_per_row=5, fig_size=(10,10), show_labels=True):
        '''
        Displays sample of the training dataset.
        '''
        return show_dataloader_sample(dataloader=self.train_data_loader, num_of_images_per_row=num_of_images_per_row, figsize=fig_size, show_labels=show_labels)

    def train(self, verbose=True):
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
                                                device = self.device,
                                                verbose=verbose)

    def metrics(self):
        '''
        Display the training metrics.
        '''
        show_metrics(self.train_metrics)

    def export_model(self,output_path):
        '''
        Exports the trained model into a target file.
        '''
        torch.save(self.trained_model, output_path)
        print ('Trained classifier exported successfully.')

    def set_trained_model(self, model_path, mode):
        '''
        Loads a previously trained model into pipeline
        Inputs:
            model_path: [str] Path to target model
            mode: [str] either 'train' or 'infer'.'train' will load the model to be trained. 'infer' will load the model for inference.
        '''
        if mode == 'train':
            self.train_model = torch.load(model_path)
        elif mode == 'infer':
            self.trained_model = torch.load(model_path)
        print ('Model Loaded Successfully.')

    def inference(self, test_img_path, transformations='default'):
        '''
        Performs inference on target DICOM image using a trained classifier.
        '''
        if transformations=='default':
            transformations = self.transformations
        else:
            transformations = transformations

        pred, percent = model_inference(model=self.trained_model,input_image_path=test_img_path, inference_transformations=transformations)
        print (pred)

    def confusion_matrix(self, target_data_set='default', target_classes='default', figure_size=(8,6)):
        if target_data_set=='default':
            target_data_set = self.valid_data_set
        else:
            target_data_set = target_data_set

        if target_classes == 'default':
            target_classes = self.data_set.classes
        else:
            target_classes = target_classes

        show_confusion_matrix(model=self.trained_model, target_data_set=target_data_set, target_classes=target_classes, figure_size=figure_size)

    def roc(self, target_data_set='default', auc=True, figure_size=(10,10)):
        if target_data_set=='default':
            target_data_set = self.valid_data_set
        else:
            target_data_set = target_data_set

        show_nn_roc(model=self.trained_model, target_data_set=target_data_set, auc=auc, figure_size=figure_size)






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
