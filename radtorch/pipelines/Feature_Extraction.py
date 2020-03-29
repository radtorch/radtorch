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



class Feature_Extraction():

    '''
    .. include:: ./documentation/docs/pipeline.md##Feature_Extraction
    '''

    def __init__(
    self,
    data_directory,
    transformations='default',
    custom_resize = 'default',
    device='default',
    is_dicom=True,
    label_from_table=False,
    is_csv=None,
    table_source=None,
    path_col = 'IMAGE_PATH',
    label_col = 'IMAGE_LABEL' ,
    mode='RAW',
    wl=None,
    model_arch='vgg16',
    pre_trained=True,
    batch_size=16,
    unfreeze_weights=False,
    shuffle=True
    ):
        self.data_directory = data_directory
        self.label_from_table = label_from_table
        self.is_csv = is_csv
        self.is_dicom = is_dicom
        self.table_source = table_source
        self.mode = mode
        self.wl = wl
        self.batch_size=batch_size
        self.shuffle=shuffle

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


        self.model_arch = model_arch
        self.pre_trained = pre_trained
        self.unfreeze_weights = unfreeze_weights
        self.path_col = path_col
        self.label_col = label_col
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

        self.data_loader = torch.utils.data.DataLoader(
                                                    self.data_set,
                                                    batch_size=self.batch_size,
                                                    shuffle=self.shuffle,
                                                    num_workers=4)


        self.num_output_classes = len(self.data_set.classes)

        self.model = create_model(
                                    model_arch=self.model_arch,
                                    output_classes=self.num_output_classes,
                                    pre_trained=self.pre_trained,
                                    unfreeze_weights = self.unfreeze_weights,
                                    mode = 'feature_extraction',
                                    )

        self.model = self.model.to(self.device)

    def info(self):
        '''
        Displays Feature Extraction Pipeline Parameters.
        '''
        print ('RADTorch Feature Extraction Pipeline Parameters')
        info = {key:str(value) for key, value in self.__dict__.items()}
        extractor_info = pd.DataFrame.from_dict(info.items())
        extractor_info.columns = ['Property', 'Value']
        return extractor_info

    def dataset_info(self, plot=False):
        '''
        Displays Dataset Information.
        '''

        info = {}

        info['data_set'] = show_dataset_info(self.data_set)

        if plot:
            plot_dataset_info(info_dict, plot_size= plot_size)
        else:
            display (show_dataset_info(self.data_set))

    def sample(self, fig_size=(10,10), show_labels=True, show_file_name=False):
        '''
        Display sample of the training dataset.
        Inputs:
            num_of_images_per_row: _(int)_ number of images per column. (default=5)
            fig_size: _(tuple)_figure size. (default=(10,10))
            show_labels: _(boolean)_ show the image label idx. (default=True)
        '''
        batch = next(iter(self.data_loader))
        images, labels, paths = batch
        images = images.numpy()
        images = [np.moveaxis(x, 0, -1) for x in images]
        if show_labels:
          titles = labels.numpy()
          titles = [((list(self.data_set.class_to_idx.keys())[list(self.data_set.class_to_idx.values()).index(i)]), i) for i in titles]
        if show_file_name:
          titles = [ntpath.basename(x) for x in paths]
        plot_images(images=images, titles=titles, figure_size=fig_size)

    def num_features(self):
        output = model_dict[self.model_arch]['output_features']
        return output

    def run(self, verbose=True):
        '''
        Extract features from the dataset
        '''
        self.features = []
        self.labels_idx = []
        self.img_path_list = []

        self.model = self.model.to(self.device)

        for i, (imgs, labels, paths) in tqdm(enumerate(self.data_loader), total=len(self.data_loader)):
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

    def export(self, target_path):
        '''
        Exports the whole image classification pipelie for future use

        ***Arguments**
        - target_path: _(str)_ target location for export.
        '''
        outfile = open(target_path,'wb')
        pickle.dump(self,outfile)
        outfile.close()

    def plot_extracted_features(self, feature_table=None, feature_names=None, num_features=100, num_images=100,image_path_col='img_path', image_label_col='label_idx'):
        if feature_table == None:
            feature_table = self.feature_table
        if feature_names == None:
            feature_names = self.feature_names
        return plot_features(feature_table, feature_names, num_features, num_images,image_path_col, image_label_col)




if __name__ == "__main__":
    Feature_Extraction()
