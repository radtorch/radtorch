# Copyright (C) 2020 RADTorch and Mohamed Elbanan, MD
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see https://www.gnu.org/licenses/



"""
Functions and Classes RADTorch Pipelines
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


from radtorch.modelsutils import *
from radtorch.datautils import *
from radtorch.visutils import *
from radtorch.generalutils import *




class Pipeline():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        for K, V in DEFAULTS_PIPELINE_SETTINGS.items():
            if K not in kwargs.keys():
                setattr(self, K, V)

        if 'custom_resize' not in kwargs.keys():
            self.input_resize = model_dict[self.model_arch]['input_size']

        if 'transformations' not in kwargs.keys():
            if self.is_dicom:
                self.transformations = transforms.Compose([
                        transforms.Resize((self.input_resize, self.input_resize)),
                        transforms.transforms.Grayscale(3),
                        transforms.ToTensor()])
            else:
                self.transformations = transforms.Compose([
                        transforms.Resize((self.input_resize, self.input_resize)),
                        transforms.ToTensor()])

        if 'device' not in kwargs.keys():
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # Load predefined tables or Create Master Dataset and dataloaders
        if self.load_predefined_datatables:
            self.train_dataset, self.valid_dataset, self.test_dataset = load_predefined_datatables(data_directory=self.data_directory,is_csv=self.is_csv,is_dicom=self.is_dicom,predefined_datasets=self.predefined_datasets,path_col=self.path_col,label_col=self.label_col,mode=self.mode,wl=self.wl,transformations=self.transformations )

        # Load predefined tables if available
        else: # Else create master dataset
            if self.label_from_table == True:
                try:
                    self.dataset = dataset_from_table(data_directory=self.data_directory,is_csv=self.is_csv,is_dicom=self.is_dicom,input_source=self.table_source,img_path_column=self.path_col,img_label_column=self.label_col,multi_label = self.multi_label,mode=self.mode,wl=self.wl,trans=self.transformations)
                except:
                    raise TypeError('Dataset could not be created from table.')
                    pass #Create Master Dataset from Table
            if self.label_from_table == False:
                try:
                    self.dataset = dataset_from_folder(data_directory=self.data_directory,is_dicom=self.is_dicom,mode=self.mode,wl=self.wl,trans=self.transformations)
                except:
                    raise TypeError('Dataset could not be created from folder structure.')
                    pass #Create Master Dataset from Folder

            self.train_dataset, self.valid_dataset, self.test_dataset = split_dataset(dataset=self.data_set, valid_percent=self.valid_percent, test_percent=self.test_percent, equal_class_split=True, shuffle=True)
            if self.balance_class:
                self.train_dataset = over_sample(self.train_dataset)
                self.valid_dataset = over_sample(self.valid_dataset)
                if len(self.test_dataset)>0:self.test_dataset = over_sample(self.test_dataset)
            self.num_output_classes = len(self.data_set.classes)

        # DataLoaders
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers)
        self.valid_dataloader = torch.utils.data.DataLoader(self.valid_dataset,batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers)
        if len(self.test_dataset)>0: self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset,batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers)

        if self.normalize == 'auto':
            self.mean, self.std = calculate_mean_std(torch.utils.data.DataLoader(self.dataset,batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers))
        elif type(self.normalize) is list:
            self.mean = self.normalize[0]
            self.std = self.normalize[1]

        self.train_model = create_model(model_arch=self.model_arch,output_classes=self.num_output_classes,pre_trained=self.pre_trained,unfreeze_weights = self.unfreeze_weights,mode = 'train',)

        self.train_model = self.train_model.to(self.device)



    def info(self):
        info = pd.DataFrame.from_dict(self.__dict__.items())
        info.columns = ['Property', 'Value']
        info = info.append({'Property':'Train Dataset Size', 'Value':len(self.train_dataset)}, ignore_index=True)
        info = info.append({'Property':'Valid Dataset Size', 'Value':len(self.valid_dataset)}, ignore_index=True)
        if self.test_percent > 0:
            info = info.append({'Property':'Test Dataset Size', 'Value':len(self.test_dataset)}, ignore_index=True)
        return info

    def dataset_info(self, plot=True, fig_size=(500,300)):
        info_dict = {}
        info_dict['train_dataset'] = show_dataset_info(self.train_dataset)
        info_dict['train_dataset'].style.set_caption("train_dataset")
        info_dict['valid_dataset'] = show_dataset_info(self.valid_dataset)
        info_dict['valid_dataset'].style.set_caption("valid_dataset")
        if self.test_percent > 0:
            info_dict['test_dataset'] = show_dataset_info(self.test_dataset)
            info_dict['test_dataset'].style.set_caption("test_dataset")

        if plot:
            plot_dataset_info(info_dict, plot_size= plot_size)
        else:

            display (show_dataset_info(self.train_dataset))
            display (show_dataset_info(self.valid_dataset))
            display (show_dataset_info(self.test_dataset))

    def sample(self, fig_size=(10,10), show_labels=True, show_file_name=False):
        batch = next(iter(self.train_dataloader))
        images, labels, paths = batch
        images = [np.moveaxis(x, 0, -1) for x in images.numpy()]
        if show_labels:
          titles = labels.numpy()
          titles = [((list(self.train_dataset.class_to_idx.keys())[list(self.train_dataset.class_to_idx.values()).index(i)]), i) for i in titles]
        if show_file_name:
          titles = [ntpath.basename(x) for x in paths]
        plot_images(images=images, titles=titles, figure_size=fig_size)

    def metrics(self, fig_size=(500,300)):
        return show_metrics(self.classifiers,  fig_size=fig_size)

    def export(self, output_path):
        outfile = open(output_path,'wb')
        pickle.dump(self,outfile)
        outfile.close()




class Image_Classification(Pipeline):
    def __init__(self, **kwargs):
        Pipeline.__init__(self)

        if self.loss_function in supported_image_classification_losses:
            self.loss_function = create_loss_function(self.loss_function)
        else:
            raise TypeError('Selected loss function is not supported with image classification pipeline. Please use modelsutils.supported() to view list of supported loss functions.')
            pass

        if optimizer in supported_optimizer:
            self.optimizer = create_optimizer(traning_model=self.train_model, optimizer_type=optimizer, learning_rate=self.learning_rate)
        else:
            raise TypeError('Selected optimizer is not supported with image classification pipeline. Please use modelsutils.supported() to view list of supported optimizers.')
            pass


    def run(self, verbose=True):

        try:
            print ('Starting Image Classification Pipeline Training')
            self.trained_model, self.train_metrics = train_model(
                                                    model = self.train_model,
                                                    train_data_loader = self.train_dataloader,
                                                    valid_data_loader = self.valid_dataloader,
                                                    train_data_set = self.train_dataset,
                                                    valid_data_set = self.valid_dataset,
                                                    loss_criterion = self.loss_function,
                                                    optimizer = self.optimizer,
                                                    epochs = self.train_epochs,
                                                    device = self.device,
                                                    verbose=verbose)
            self.train_metrics = pd.DataFrame(data=self.train_metrics, columns = ['Train_Loss', 'Valid_Loss', 'Train_Accuracy', 'Valid_Accuracy'])
        except:
            raise TypeError('Could not train image classification pipeline. Please check provided parameters.')
            pass
