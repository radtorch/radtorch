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

        for K, V in self.DEFAULT_SETTINGS.items():
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
            self.train_dataset, self.valid_dataset, self.test_dataset = load_predefined_datatables(data_directory=self.data_directory,is_csv=self.is_csv,is_dicom=self.is_dicom,predefined_datasets=self.load_predefined_datatables,path_col=self.path_col,label_col=self.label_col,mode=self.mode,wl=self.wl,transformations=self.transformations )
            self.num_output_classes = len(self.train_dataset.classes)

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

            self.train_dataset, self.valid_dataset, self.test_dataset = split_dataset(dataset=self.dataset, valid_percent=self.valid_percent, test_percent=self.test_percent, equal_class_split=True, shuffle=True)
            if self.balance_class:
                self.train_dataset = over_sample(self.train_dataset)
                self.valid_dataset = over_sample(self.valid_dataset)
                if len(self.test_dataset)>0:self.test_dataset = over_sample(self.test_dataset)
            self.num_output_classes = len(self.dataset.classes)

        # DataLoaders
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers)
        self.valid_dataloader = torch.utils.data.DataLoader(self.valid_dataset,batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers)
        if len(self.test_dataset)>0: self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset,batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers)

        if self.normalize == 'auto':
            self.mean, self.std = calculate_mean_std(torch.utils.data.DataLoader(self.dataset,batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers))
        elif type(self.normalize) is list:
            self.mean = self.normalize[0]
            self.std = self.normalize[1]


    def info(self):
        info = {key:str(value) for key, value in self.__dict__.items()}
        info = pd.DataFrame.from_dict(info.items())
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
            plot_dataset_info(info_dict, plot_size= fig_size)
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
        try:
            outfile = open(output_path,'wb')
            pickle.dump(self,outfile)
            outfile.close()
            print ('Pipeline exported successfully.')
        except:
            raise TypeError('Error! Pipeline could not be exported.')

def load_pipeline(target_path):
    '''
    .. include:: ./documentation/docs/pipeline.md##load_pipeline
    '''

    infile = open(target_path,'rb')
    pipeline = pickle.load(infile)
    infile.close()

    return pipeline

class Image_Classification(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(DEFAULT_SETTINGS=IMAGE_CLASSIFICATION_PIPELINE_SETTINGS, **kwargs)
        self.classifiers = [self]

        self.train_model = create_model(model_arch=self.model_arch,output_classes=self.num_output_classes,pre_trained=self.pre_trained,unfreeze_weights = self.unfreeze_weights,mode = 'train',)

        self.train_model = self.train_model.to(self.device)


        if self.loss_function in supported_image_classification_losses:
            self.loss_function = create_loss_function(self.loss_function)
        else:
            raise TypeError('Selected loss function is not supported with image classification pipeline. Please use modelsutils.supported() to view list of supported loss functions.')
            pass

        if self.optimizer in supported_optimizer:
            self.optimizer = create_optimizer(traning_model=self.train_model, optimizer_type=self.optimizer, learning_rate=self.learning_rate)
        else:
            raise TypeError('Selected optimizer is not supported with image classification pipeline. Please use modelsutils.supported() to view list of supported optimizers.')
            pass

    def classes(self):
        return self.train_dataset.class_to_idx

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

    def export_model(self,output_path):
        try:
            torch.save(self.trained_model, output_path)
            print ('Trained classifier exported successfully.')
        except:
            raise TypeError('Error! Trained Model could not be exported.')

    def inference(self, transformations=None, all_predictions=False, *args, **kwargs):
        if transformations==None:
            transformations=self.transformations
        return model_inference( model=self.trained_model,
                                input_image_path=target_image_path,
                                inference_transformations=transformations,
                                all_predictions=all_predictions)

    def confusion_matrix(self, figure_size=(7,7), target_dataset = None, target_classes = None, cmap=None, *args,  **kwargs):
        if target_dataset==None:
            target_dataset=self.test_dataset
        if target_classes==None:
            target_classes=self.dataset.classes

        target_dataset.trans = self.transformations
        show_nn_confusion_matrix(model=self.trained_model, target_data_set=target_dataset, target_classes=target_classes, figure_size=figure_size, cmap=cmap, device=self.device)

    def roc(self, target_dataset=None, figure_size=(600,400), *args,  **kwargs):
        if target_dataset==None:
            target_dataset=self.test_dataset
        num_classes = len(target_dataset.classes)
        if num_classes <= 2:
            show_roc([self], fig_size=figure_size)
        else:
            raise TypeError('ROC cannot support more than 2 classes at the current time. This will be addressed in an upcoming update.')
            pass

    def misclassified(self, target_dataset=None, num_images=16, figure_size=(10,10), show_table=False, *args,  **kwargs):
        if target_dataset==None:
            target_dataset=self.test_dataset

        target_dataset.trans = self.transformations

        self.misclassified_instances = show_nn_misclassified(model=self.trained_model, target_data_set=target_dataset, transforms=self.transformations,   is_dicom=self.is_dicom, num_of_images=num_images, device=self.device, figure_size=figure_size)

        if show_table:
            return self.misclassified_instances

class Compare_Image_Classifier():
    def __init__(self, DEFAULT_SETTINGS=COMPARE_CLASSIFIER_PIPELINE_SETTINGS, **kwargs):
        # self.DEFAULT_SETTINGS=DEFAULT_SETTINGS
        for k, v in kwargs.items():
            setattr(self, k, v)

        for K, V in DEFAULT_SETTINGS.items():
            if K not in kwargs.keys():
                setattr(self, K, V)

        self.compare_parameters = {k:v for k,v in self.__dict__.items() if type(v)==list}
        self.non_compare_parameters = {k: v for k, v in self.__dict__.items() if k not in self.compare_parameters and k !='compare_parameters'}
        self.compare_parameters_names= list(self.compare_parameters.keys())
        self.scenarios_list = list(itertools.product(*list(self.compare_parameters.values())))
        self.num_scenarios = len(self.scenarios_list)
        self.scenarios_df = pd.DataFrame(self.scenarios_list, columns =self.compare_parameters_names)

        self.classifiers = []
        for x in self.scenarios_list:
          if self.scenarios_list.index(x) == 0:
            x = list(x)
            classifier_settings = {self.compare_parameters_names[i]: x[i] for i in range(len(self.compare_parameters_names))}
            classifier_settings.update(self.non_compare_parameters)
            clf = Image_Classification(**classifier_settings)
            self.train_label_table=clf.train_dataset.input_data
            self.valid_label_table=clf.valid_dataset.input_data
            self.test_label_table=clf.test_dataset.input_data
            self.datasets = {'train':self.train_label_table, 'valid':self.valid_label_table,'test':self.test_label_table}
            self.classifiers.append(clf)
          else:
            x = list(x)
            classifier_settings = {self.compare_parameters_names[i]: x[i] for i in range(len(self.compare_parameters_names))}
            classifier_settings.update(self.non_compare_parameters)
            classifier_settings['load_predefined_datatables'] = self.datasets
            clf = Image_Classification(**classifier_settings)
            self.classifiers.append(clf)

    def grid(self):
      return self.scenarios_df

    def dataset_info(self,plot=True, figure_size=(500,300)):
        return self.classifiers[0].dataset_info(plot=plot, fig_size=figure_size)

    def sample(self, figure_size=(10,10), show_labels=True, show_file_name=False):
        return self.classifiers[0].sample(fig_size=figure_size, show_labels=show_labels, show_file_name=show_file_name)

    def classes(self):
        return self.classifiers[0].train_dataset.class_to_idx

    def parameters(self):
        return self.compare_parameters_names

    def run(self):
      self.master_metrics = []
      self.trained_models = []
      for i in tqdm(self.classifiers, total=len(self.classifiers)):
        print ('Starting Training Classifier Number',self.classifiers.index(i))
        i.run()
        self.trained_models.append(i.trained_model)
        self.master_metrics.append(i.train_metrics)
        torch.cuda.empty_cache()

    def metrics(self, figure_size=(650,400)):
        return show_metrics(self.classifiers,  fig_size=figure_size)

    def roc(self, figure_size=(700,400)):
        self.auc_list = show_roc(self.classifiers, fig_size=figure_size)
        self.best_model_auc = max(self.auc_list)
        self.best_model_index = (self.auc_list.index(self.best_model_auc))
        self.best_classifier = self.classifiers[self.best_model_index]

    def best(self, path=None, export_classifier=False, export_model=False):
        try:
            print ('Best Classifier = Model', self.best_model_index)
            print ('Best Classifier AUC =', self.best_model_auc)
            if export_model:
                export(self.best_classifier.trained_model, path)
                print (' Best Model Exported Successfully')
            if export_classifier:
                export(self.best_classifier, path)
                print (' Best Classifier Pipeline Exported Successfully')
        except:
            raise TypeError('Error! ROC and AUC for classifiers have not been estimated. Please run Compare_Image_Classifier.roc.() first')

    def export(self, output_path):
        try:
            outfile = open(output_path,'wb')
            pickle.dump(self,outfile)
            outfile.close()
            print ('Pipeline exported successfully.')
        except:
            raise TypeError('Error! Pipeline could not be exported.')

class Feature_Extraction(Pipeline):

    def __init__(self, **kwargs):
        super().__init__(DEFAULT_SETTINGS=FEATURE_EXTRACTION_PIPELINE_SETTINGS, **kwargs)
        self.classifiers = [self]

        self.model = create_model(model_arch=self.model_arch,output_classes=self.num_output_classes,pre_trained=self.pre_trained,unfreeze_weights = self.unfreeze_weights,mode = 'feature_extraction',)

        self.model = self.model.to(self.device)

    def num_features(self):
        return model_dict[self.model_arch]['output_features']

    def run(self, verbose=True):
        self.features = []
        self.labels_idx = []
        self.img_path_list = []

        self.model = self.model.to(self.device)

        for i, (imgs, labels, paths) in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)):
            self.labels_idx = self.labels_idx+labels.tolist()
            self.img_path_list = self.img_path_list+list(paths)
            with torch.no_grad():
                self.model.eval()
                imgs = imgs.to(self.device)
                output = (self.model(imgs)).tolist()
                self.features = self.features+(output)


        self.feature_names = ['f_'+str(i) for i in range(0,(model_dict[self.model_arch]['output_features']))]

        feature_table = pd.DataFrame(list(zip(self.img_path_list, self.labels_idx, self.features)), columns=['img_path','label_idx', 'features'])

        feature_table[self.feature_names] = pd.DataFrame(feature_table.features.values.tolist(), index= feature_table.index)

        feature_table = feature_table.drop(['features'], axis=1)

        print (' Features extracted successfully.')

        self.feature_table = feature_table

        if verbose:
            return self.feature_table

        self.features = self.feature_table[self.feature_names]

    def export_features(self,csv_path):
        try:
            self.feature_table.to_csv(csv_path, index=False)
            print ('Features exported to CSV successfully.')
        except:
            print ('Error! No features found. Please check again or re-run the extracion pipeline.')
            pass

    def plot_extracted_features(self, feature_table=None, feature_names=None, num_features=100, num_images=100,image_path_col='img_path', image_label_col='label_idx'):
        if feature_table == None:
            feature_table = self.feature_table
        if feature_names == None:
            feature_names = self.feature_names
        return plot_features(feature_table, feature_names, num_features, num_images,image_path_col, image_label_col)






##
