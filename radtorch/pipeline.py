
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


from radtorch.modelsutils import create_model, create_loss_function, train_model, model_inference, model_dict, create_optimizer, supported_image_classification_losses , supported_optimizer
from radtorch.datautils import dataset_from_folder, dataset_from_table
from radtorch.visutils import show_dataset_info, show_dataloader_sample, show_metrics, show_nn_confusion_matrix, show_roc, show_nn_roc, show_nn_misclassified



def load_pipeline(target_path):
    '''
    .. include:: ./documentation/docs/pipeline.md##load_pipeline
    '''

    infile = open(target_path,'rb')
    pipeline = pickle.load(infile)
    infile.close()

    return pipeline


class Image_Classification():

    '''
    .. include:: ./documentation/docs/pipeline.md##Image_Classification
    '''

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
    path_col = 'IMAGE_PATH',
    label_col = 'IMAGE_LABEL' ,
    multi_label = False ,
    mode='RAW',
    wl=None,
    batch_size=16,
    test_percent = 0.2,
    valid_percent = 0.2,
    model_arch='vgg16',
    pre_trained=True,
    unfreeze_weights=True,
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
        self.test_percent = test_percent
        self.valid_percent = valid_percent
        self.model_arch = model_arch
        self.pre_trained = pre_trained
        self.unfreeze_weights = unfreeze_weights
        self.train_epochs = train_epochs
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.path_col = path_col
        self.label_col = label_col
        self.multi_label = multi_label

        if device == 'default':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device == device

        # Create DataSet
        if self.label_from_table == True:
            try:
                self.data_set = dataset_from_table(
                        data_directory=self.data_directory,
                        is_csv=self.is_csv,
                        is_dicom=self.is_dicom,
                        input_source=self.table_source,
                        img_path_column=self.path_col,
                        img_label_column=self.label_col,
                        multi_label = self.multi_label,
                        mode=self.mode,
                        wl=self.wl,
                        trans=self.transformations)
            except:
                raise TypeError('Dataset could not be created.')
                pass

        else:
            if self.multi_label == True:
                raise TypeError('Dataset could not be created. Multilabel dataset creation is not yet allowed from folders. Please use dataset_from_table instead.')
                pass
            else:
                try:
                    self.data_set = dataset_from_folder(
                                data_directory=self.data_directory,
                                is_dicom=self.is_dicom,
                                mode=self.mode,
                                wl=self.wl,
                                trans=self.transformations)
                except:
                    raise TypeError('Dataset could not be created.')
                    pass


        valid_size = int(self.valid_percent*len(self.data_set))
        test_size = int(self.test_percent*len(self.data_set))
        train_size = len(self.data_set) - (valid_size+test_size)

        if self.test_percent == 0:
            self.train_data_set, self.valid_data_set = torch.utils.data.random_split(self.data_set, [train_size, valid_size])
            self.test_data_set = 0
        else:
            self.train_data_set, self.valid_data_set, self.test_data_set = torch.utils.data.random_split(self.data_set, [train_size, valid_size, test_size])

        self.train_data_loader = torch.utils.data.DataLoader(
                                                    self.train_data_set,
                                                    batch_size=self.batch_size,
                                                    shuffle=True)

        self.valid_data_loader = torch.utils.data.DataLoader(
                                                    self.valid_data_set,
                                                    batch_size=self.batch_size,
                                                    shuffle=True)

        if self.test_percent == 0:
            self.test_data_loader = 0
        else:
            self.test_data_loader = torch.utils.data.DataLoader(
                                                    self.test_data_set,
                                                    batch_size=self.batch_size,
                                                    shuffle=True)

        self.num_output_classes = len(self.data_set.classes)

        self.train_model = create_model(
                                    model_arch=self.model_arch,
                                    output_classes=self.num_output_classes,
                                    pre_trained=self.pre_trained,
                                    unfreeze_weights = self.unfreeze_weights,
                                    mode = 'train',
                                    )

        self.train_model = self.train_model.to(self.device)

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


    def info(self):
        '''
        Display Parameters of the Image Classification Pipeline.
        '''

        print ('RADTorch Image Classification Pipeline Parameters')
        for key, value in self.__dict__.items():
            if key != 'trans':
                print('>', key,'=',value)
        print ('Train Dataset Size =', len(self.train_data_set))
        print ('Valid Dataset Size =', len(self.valid_data_set))
        if self.test_percent > 0:
            print ('Test Dataset Size =', len(self.test_data_set))

    def dataset_info(self):
        '''
        Display Dataset Information.
        '''
        info = show_dataset_info(self.data_set)
        info = info.append({'Classes':'Train Dataset Size', 'Class Idx': '','Number of Instances':len(self.train_data_set)}, ignore_index=True )
        info = info.append({'Classes':'Valid Dataset Size', 'Class Idx': '','Number of Instances':len(self.valid_data_set)}, ignore_index=True )

        if self.test_percent > 0:
            info = info.append({'Classes':'Test Dataset Size', 'Class Idx': '', 'Number of Instances':len(self.test_data_set)}, ignore_index=True )

        return info

    def sample(self, num_of_images_per_row=4, fig_size=(10,10), show_labels=False):
        '''
        Display sample of the training dataset.
        Inputs:
            num_of_images_per_row: _(int)_ number of images per column. (default=5)
            fig_size: _(tuple)_figure size. (default=(10,10))
            show_labels: _(boolean)_ show the image label idx. (default=True)
        '''
        return show_dataloader_sample(dataloader=self.train_data_loader, num_of_images_per_row=num_of_images_per_row, figsize=fig_size, show_labels=show_labels)

    def train(self, verbose=True):
        '''
        Train the image classification pipeline.
        Inputs:
            verbose: _(boolean)_ Show display progress after each epoch. (default=True)
        '''
        try:
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
        except:
            raise TypeError('Could not train image classification pipeline. Please check rpovided parameters.')
            pass

    def metrics(self, fig_size=(15,5)):
        '''
        Display the training metrics.
        '''
        show_metrics(self.train_metrics, fig_size=fig_size)

    def export_model(self,output_path):
        '''
        Export the trained model into a target file.
        Inputs:
            output_path: _(str)_ path to output file. For example 'foler/folder/model.pth'
        '''

        torch.save(self.trained_model, output_path)
        print ('Trained classifier exported successfully.')

    def set_trained_model(self, model_path, mode):
        '''
        Loads a previously trained model into pipeline
        Inputs:
            model_path: _(str)_ path to target model
            mode: _(str)_ either 'train' or 'infer'.'train' will load the model to be trained. 'infer' will load the model for inference.
        '''

        if mode == 'train':
            self.train_model = torch.load(model_path)
        elif mode == 'infer':
            self.trained_model = torch.load(model_path)
        print ('Model Loaded Successfully.')

    def inference(self, test_img_path, transformations='default'):
        '''
        Performs inference on target DICOM image using a trained classifier.
        Inputs:
            test_img_path: _(str)_ path to target image.
            transformations: _(pytorch transforms list)_ pytroch transforms to be performed on the target image. (default='default')
        Outputs:
            Output: _(tuple)_ tuple of prediction class idx and accuracy percentage.
        '''
        if transformations=='default':
            transformations = self.transformations
        else:
            transformations = transformations

        pred, percent = model_inference(model=self.trained_model,input_image_path=test_img_path, inference_transformations=transformations)
        print (pred)
        return (pred, percent)

    def confusion_matrix(self, target_data_set='default', target_classes='default', figure_size=(7,7), cmap=None):
        '''
        Display Confusion Matrix
        Inputs:
            target_data_set: _(pytorch dataset object)_ dataset used for predictions to create the confusion matrix. By default, the image classification pipeline uses the test dataset created to calculate the matrix.
            target_classes: _(list)_ list of classes. By default, the image classification pipeline uses the training classes.
            figure_size: _(tuple)_figure size. (default=(7,7))
        '''

        if target_data_set=='default':
            if self.test_data_set == 0:
                raise TypeError('Error. Test Percent set to Zero in image classification pipeline. Please change or set another target testing dataset.')
                pass
            else:
                target_data_set = self.test_data_set
        else:
            target_data_set = target_data_set
            target_data_set.trans = self.transformations

        if target_classes == 'default':
            target_classes = self.data_set.classes
        else:
            target_classes = target_classes

        show_nn_confusion_matrix(model=self.trained_model, target_data_set=target_data_set, target_classes=target_classes, figure_size=figure_size, cmap=cmap, device=self.device)

    def roc(self, target_data_set='default', auc=True, figure_size=(7,7)):
        '''
        Display ROC and AUC
        Inputs:
            target_data_set: _(pytorch dataset object)_ dataset used for predictions to create the ROC. By default, the image classification pipeline uses the test dataset created to calculate the ROC.
            auc: _(boolen)_ Display area under curve. (default=True)
            figure_size: _(tuple)_figure size. (default=(7,7))
        '''

        if target_data_set=='default':
            if self.test_data_set == 0:
                raise TypeError('Error. Test Percent set to Zero in image classification pipeline. Please change or set another target testing dataset.')
                pass
            else:
                target_data_set = self.test_data_set
                num_classes = len(self.data_set.classes)
        else:
            target_data_set = target_data_set
            num_classes = len(target_data_set.classes)
            target_data_set.trans = self.transformations

        if num_classes <= 2:
            show_nn_roc(model=self.trained_model, target_data_set=target_data_set, auc=auc, figure_size=figure_size, device=self.device)
        else:
            raise TypeError('ROC cannot support more than 2 classes at the current time. This will be addressed in an upcoming update.')
            pass

    def misclassified(self, target_data_set='default', num_of_images=16, figure_size=(7,7), show_table=False):
        if target_data_set=='default':
            if self.test_data_set == 0:
                raise TypeError('Error. Test Percent set to Zero in image classification pipeline. Please change or set another target testing dataset.')
                pass
            else:
                target_data_set = self.test_data_set
        else:
            target_data_set = target_data_set
            target_data_set.trans = self.transformations

        self.misclassified_instances = show_nn_misclassified(model=self.trained_model, target_data_set=target_data_set, is_dicom=self.is_dicom, num_of_images=num_of_images, device=self.device, figure_size=figure_size)

        if show_table:
            self.misclassified_instances
        return self.misclassified_instances

    def export(self, target_path):
        '''
        Exports the whole image classification pipelie for future use

        ***Arguments**
        - target_path: _(str)_ target location for export.
        '''
        outfile = open(target_path,'wb')
        pickle.dump(self,outfile)
        outfile.close()


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
                                                    shuffle=self.shuffle)


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
        for key, value in self.__dict__.items():
            if key != 'trans':
                print('>', key,'=',value)


    def dataset_info(self):
        '''
        Displays Dataset Information.
        '''
        return (show_dataset_info(self.data_set))


    def sample(self, num_of_images_per_row=5, fig_size=(10,10), show_labels=True):
        '''
        Displays sample of the dataset.
        '''
        return show_dataloader_sample(dataloader=self.data_loader, num_of_images_per_row=num_of_images_per_row, figsize=fig_size, show_labels=show_labels)


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
            self.feature_table

        self.features = self.feature_table[self.feature_names]

        return self.feature_table


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

    # def set_trained_model(self, model_path, mode):
    #     '''
    #     Loads a previously trained model into pipeline
    #     Inputs:
    #         model_path: [str] Path to target model
    #         mode: [str] either 'train' or 'infer'.'train' will load the model to be trained. 'infer' will load the model for inference.
    #     '''
    #     if mode == 'train':
    #         self.model = torch.load(model_path)
    #     elif mode == 'infer':
    #         self.model = torch.load(model_path)
    #
    #     print ('Model Loaded Successfully.')
