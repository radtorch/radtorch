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

from radtorch.modelsutils import create_model, create_loss_function, train_model, model_inference, model_dict, create_optimizer, supported_image_classification_losses , supported_optimizer
from radtorch.datautils import dataset_from_folder, dataset_from_table, split_dataset, calculate_mean_std, over_sample
from radtorch.visutils import show_dataset_info, show_dataloader_sample, show_metrics, show_nn_confusion_matrix, show_roc, show_nn_misclassified, plot_features, plot_pipline_dataset_info, plot_images ,plot_dataset_info
from radtorch.generalutils import export

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
    name = None,
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
    balance_class = False,
    multi_label = False,
    predefined_datasets = False,
    mode='RAW',
    wl=None,
    normalize='default',
    batch_size=16,
    test_percent = 0.2,
    valid_percent = 0.2,
    model_arch='vgg16',
    pre_trained=True,
    unfreeze_weights=True,
    train_epochs=20,
    learning_rate=0.0001,
    loss_function='CrossEntropyLoss'):
        self.name = name
        self.data_directory = data_directory
        self.label_from_table = label_from_table
        self.is_csv = is_csv
        self.is_dicom = is_dicom
        self.table_source = table_source
        self.mode = mode
        self.wl = wl
        self.normalize = normalize
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
        self.num_workers = 0
        self.balance_class = balance_class
        self.predefined_datasets = predefined_datasets


        # Custom Resize
        if custom_resize=='default':
            self.input_resize = model_dict[model_arch]['input_size']
        else:
            self.input_resize = custom_resize

        # Transformations
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

        # Device
        if device == 'default':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device == device


        if self.predefined_datasets:
            self.train_data_set = dataset_from_table(
                                                    data_directory=self.data_directory,
                                                    is_csv=self.is_csv,
                                                    is_dicom=self.is_dicom,
                                                    input_source=self.predefined_datasets['train'],
                                                    img_path_column=self.path_col,
                                                    img_label_column=self.label_col,
                                                    multi_label = self.multi_label,
                                                    mode=self.mode,
                                                    wl=self.wl,
                                                    trans=self.transformations)

            self.valid_data_set = dataset_from_table(
                                                    data_directory=self.data_directory,
                                                    is_csv=self.is_csv,
                                                    is_dicom=self.is_dicom,
                                                    input_source=self.predefined_datasets['valid'],
                                                    img_path_column=self.path_col,
                                                    img_label_column=self.label_col,
                                                    multi_label = self.multi_label,
                                                    mode=self.mode,
                                                    wl=self.wl,
                                                    trans=self.transformations)

            self.test_data_set = dataset_from_table(
                                                    data_directory=self.data_directory,
                                                    is_csv=self.is_csv,
                                                    is_dicom=self.is_dicom,
                                                    input_source=self.predefined_datasets['test'],
                                                    img_path_column=self.path_col,
                                                    img_label_column=self.label_col,
                                                    multi_label = self.multi_label,
                                                    mode=self.mode,
                                                    wl=self.wl,
                                                    trans=self.transformations)

            self.num_output_classes = len(self.train_data_set.classes)
            self.train_data_loader = torch.utils.data.DataLoader(
                                                    self.train_data_set,
                                                    batch_size=self.batch_size,
                                                    shuffle=True,
                                                    num_workers=self.num_workers)
            self.valid_data_loader = torch.utils.data.DataLoader(
                                                    self.valid_data_set,
                                                    batch_size=self.batch_size,
                                                    shuffle=True,
                                                    num_workers=self.num_workers)
            self.test_data_loader = torch.utils.data.DataLoader(
                                                    self.test_data_set,
                                                    batch_size=self.batch_size,
                                                    shuffle=True,
                                                    num_workers=self.num_workers)


        else:


            # Create Dataset
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
                    raise TypeError('Dataset could not be created from table.')
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
                    raise TypeError('Dataset could not be created from folder structure.')
                    pass

            if self.normalize:
                if self.normalize == 'auto':
                    self.data_loader = torch.utils.data.DataLoader(
                                                            self.data_set,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            num_workers=self.num_workers)

                    self.mean, self.std = calculate_mean_std(self.data_loader)
                elif self.normalize == 'default':
                    self.mean = [0.5, 0.5, 0.5]
                    self.std = [0.5, 0.5, 0.5]
                else:
                    self.mean = self.normalize[0]
                    self.std = self.normalize[1]

                if transformations == 'default':
                    if self.is_dicom == True:
                        self.transformations = transforms.Compose([
                                transforms.Resize((self.input_resize, self.input_resize)),
                                transforms.transforms.Grayscale(3),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=self.mean, std=self.std)])
                    else:
                        self.transformations = transforms.Compose([
                                transforms.Resize((self.input_resize, self.input_resize)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=self.mean, std=self.std)])
                else:
                    self.transformations = transformations

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
                        raise TypeError('Dataset could not be created from table.')
                        pass


            # Split Data set
            if self.test_percent == 0:
                self.train_data_set, self.valid_data_set = split_dataset(dataset=self.data_set, valid_percent=self.valid_percent, test_percent=self.test_percent, equal_class_split=True, shuffle=True)
                if self.balance_class:
                    self.train_data_set = over_sample(self.train_data_set)
                    self.valid_data_set = over_sample(self.valid_data_set)
                self.test_data_set = 0
            else:
                self.train_data_set, self.valid_data_set, self.test_data_set = split_dataset(dataset=self.data_set, valid_percent=self.valid_percent, test_percent=self.test_percent, equal_class_split=True, shuffle=True)
                if self.balance_class:
                    self.train_data_set = over_sample(self.train_data_set)
                    self.valid_data_set = over_sample(self.valid_data_set)
                    self.test_data_set = over_sample(self.test_data_set)

            self.num_output_classes = len(self.data_set.classes)

        # Data Loaders
            self.train_data_loader = torch.utils.data.DataLoader(
                                                        self.train_data_set,
                                                        batch_size=self.batch_size,
                                                        shuffle=True,
                                                        num_workers=self.num_workers)


            self.valid_data_loader = torch.utils.data.DataLoader(
                                                        self.valid_data_set,
                                                        batch_size=self.batch_size,
                                                        shuffle=True,
                                                        num_workers=self.num_workers)


            if self.test_percent == 0:
                self.test_data_loader = 0
            else:
                self.test_data_loader = torch.utils.data.DataLoader(
                                                        self.test_data_set,
                                                        batch_size=self.batch_size,
                                                        shuffle=True,
                                                        num_workers=self.num_workers)

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
        info = {key:str(value) for key, value in self.__dict__.items()}
        classifier_info = pd.DataFrame.from_dict(info.items())
        classifier_info.columns = ['Property', 'Value']

        classifier_info = classifier_info.append({'Property':'Train Dataset Size', 'Value':len(self.train_data_set)}, ignore_index=True)
        classifier_info = classifier_info.append({'Property':'Valid Dataset Size', 'Value':len(self.valid_data_set)}, ignore_index=True)

        if self.test_percent > 0:
            classifier_info = classifier_info.append({'Property':'Test Dataset Size', 'Value':len(self.test_data_set)}, ignore_index=True)

        return classifier_info

    def dataset_info(self, plot=True, plot_size=(500,300)):
        '''
        Display Dataset Information.
        '''

        info_dict = {}

        # # Display the train/valid/test size
        # master_dataset_info = pd.DataFrame()
        # master_dataset_info = master_dataset_info.append({'Classes':'Train Dataset Size','Number of Instances':len(self.train_data_set)}, ignore_index=True )
        # master_dataset_info = master_dataset_info.append({'Classes':'Valid Dataset Size','Number of Instances':len(self.valid_data_set)}, ignore_index=True )
        #
        # if self.test_percent > 0:
        #     master_dataset_info = master_dataset_info.append({'Classes':'Test Dataset Size', 'Class Idx': '', 'Number of Instances':len(self.test_data_set)}, ignore_index=True )
        # else:
        #     self.test_percent = []
        #
        # master_dataset_info = master_dataset_info.append({'Classes':'Full Dataset Size','Number of Instances':(len(self.train_data_set)+len(self.valid_data_set)+len(self.test_data_set))}, ignore_index=True )
        # info_dict['full_dataset'] = master_dataset_info


        # Display train breakdown by class
        info_dict['train_dataset'] = show_dataset_info(self.train_data_set)

        # Display valid breakdown by class
        info_dict['valid_dataset'] = show_dataset_info(self.valid_data_set)

        # Display test breakdown by class
        if self.test_percent > 0:
            info_dict['test_dataset'] = show_dataset_info(self.test_data_set)



        if plot:
            plot_dataset_info(info_dict, plot_size= plot_size)
            # plot_pipline_dataset_info(info, test_percent = self.test_percent)
        else:
            return info

    def sample(self, fig_size=(10,10), show_labels=True, show_file_name=False):
        '''
        Display sample of the training dataset.
        Inputs:
            num_of_images_per_row: _(int)_ number of images per column. (default=5)
            fig_size: _(tuple)_figure size. (default=(10,10))
            show_labels: _(boolean)_ show the image label idx. (default=True)
        '''
        # return show_dataloader_sample(dataloader=self.train_data_loader, num_of_images_per_row=num_of_images_per_row, figsize=fig_size, show_labels=show_labels)
        batch = next(iter(self.train_data_loader))
        images, labels, paths = batch
        images = images.numpy()
        images = [np.moveaxis(x, 0, -1) for x in images]
        if show_labels:
          titles = labels.numpy()
          titles = [((list(self.data_set.class_to_idx.keys())[list(self.data_set.class_to_idx.values()).index(i)]), i) for i in titles]
        if show_file_name:
          titles = [ntpath.basename(x) for x in paths]
        plot_images(images=images, titles=titles, figure_size=fig_size)

    def run(self, verbose=True):
        '''
        Train the image classification pipeline.
        Inputs:
            verbose: _(boolean)_ Show display progress after each epoch. (default=True)
        '''
        try:
            print ('Starting Image Classification Pipeline Training')
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
            self.train_metrics = pd.DataFrame(data=self.train_metrics, columns = ['Train_Loss', 'Valid_Loss', 'Train_Accuracy', 'Valid_Accuracy'])
        except:
            raise TypeError('Could not train image classification pipeline. Please check rpovided parameters.')
            pass

    def metrics(self, fig_size=(500,300)):
        '''
        Display the training metrics.
        '''
        # show_metrics(self.train_metrics, fig_size=fig_size)
        # show_metrics(self.train_metrics, metric=metrics, show_points = show_points, fig_size = fig_size)
        show_metrics([self], fig_size=(fig_size))

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

    def inference(self, test_img_path, transformations='default',  all_predictions=False):
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

        return model_inference(model=self.trained_model,input_image_path=test_img_path, inference_transformations=transformations, all_predictions=all_predictions)

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

    def roc(self, target_data_set='default', figure_size=(600,400)):
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
            show_roc([self], fig_size=figure_size)
        else:
            raise TypeError('ROC cannot support more than 2 classes at the current time. This will be addressed in an upcoming update.')
            pass

    def misclassified(self, target_data_set='default', num_of_images=16, figure_size=(10,10), show_table=False):
        if target_data_set=='default':
            if self.test_data_set == 0:
                raise TypeError('Error. Test Percent set to Zero in image classification pipeline. Please change or set another target testing dataset.')
                pass
            else:
                target_data_set = self.test_data_set
        else:
            target_data_set = target_data_set
            target_data_set.trans = self.transformations

        self.misclassified_instances = show_nn_misclassified(model=self.trained_model, target_data_set=target_data_set, transforms=self.transformations,   is_dicom=self.is_dicom, num_of_images=num_of_images, device=self.device, figure_size=figure_size)

        if show_table:
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
        info = show_dataset_info(self.data_set)

        if plot:
            plot_pipline_dataset_info(info, test_percent = 0)
        else:
            return info

    def sample(self, fig_size=(10,10), show_labels=True, show_file_name=False):
        '''
        Display sample of the training dataset.
        Inputs:
            num_of_images_per_row: _(int)_ number of images per column. (default=5)
            fig_size: _(tuple)_figure size. (default=(10,10))
            show_labels: _(boolean)_ show the image label idx. (default=True)
        '''
        # return show_dataloader_sample(dataloader=self.train_data_loader, num_of_images_per_row=num_of_images_per_row, figsize=fig_size, show_labels=show_labels)
        batch = next(iter(self.train_data_loader))
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


class Compare_Image_Classifier():

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
    balance_class =[False],
    multi_label = False,
    mode='RAW',
    wl=None,
    normalize=['default'],
    batch_size=[8],
    test_percent = [0.2],
    valid_percent = [0.2],
    model_arch=['vgg16'],
    pre_trained=[True],
    unfreeze_weights=True,
    train_epochs=[10],
    learning_rate=[0.0001],
    loss_function='CrossEntropyLoss'):
        self.data_directory = data_directory
        self.transformations = transformations
        self.custom_resize = custom_resize
        self.device = device
        self.optimizer = optimizer
        self.label_from_table = label_from_table
        self.is_dicom = is_dicom
        self.is_csv = is_csv
        self.table_source = table_source
        self.path_col = path_col
        self.label_col = label_col
        self.balance_class = balance_class
        self.multi_label = multi_label
        self.mode = mode
        self.wl = wl
        self.normalize = normalize
        self.batch_size = batch_size
        self.test_percent = test_percent
        self.valid_percent = valid_percent
        self.model_arch = model_arch
        self.pre_trained = pre_trained
        self.unfreeze_weights = unfreeze_weights
        self.train_epochs = train_epochs
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.num_workers = 0

        variables = [
        self.balance_class,
        self.normalize,
        self.batch_size,
        self.test_percent,
        self.valid_percent,
        self.train_epochs,
        self.learning_rate,
        self.model_arch,
        self.pre_trained
        ]

        self.variables_names = ['balance_class', 'normalize', 'batch_size', 'test_percent','valid_percent','train_epochs','learning_rate', 'model_arch','pre_trained']

        self.scenarios_list = list(itertools.product(*variables))
        self.num_scenarios = len(self.scenarios_list)
        self.scenarios_df = pd.DataFrame(self.scenarios_list, columns =self.variables_names)

        self.classifiers = []

        for i in self.scenarios_list:
            balance_class = i[0]
            normalize = i[1]
            batch_size = i[2]
            test_percent = i[3]
            valid_percent = i[4]
            train_epochs = i[5]
            learning_rate = i[6]
            model_arch = i[7]
            pre_trained  = i[8]

            if self.scenarios_list.index(i) == 0:
                clf = Image_Classification(data_directory = self.data_directory,
                                                      name = None,
                                                      transformations=self.transformations,
                                                      custom_resize = self.custom_resize,
                                                      device=self.device,
                                                      optimizer=self.optimizer,
                                                      is_dicom=self.is_dicom,
                                                      label_from_table=self.label_from_table,
                                                      is_csv=self.is_csv,
                                                      table_source=self.table_source,
                                                      path_col = self.path_col,
                                                      label_col = self.label_col ,
                                                      balance_class = balance_class,
                                                      multi_label = self.multi_label,
                                                      mode=self.mode,
                                                      wl=self.wl,
                                                      normalize=normalize,
                                                      batch_size=batch_size,
                                                      test_percent = test_percent,
                                                      valid_percent = valid_percent,
                                                      model_arch=model_arch,
                                                      pre_trained=pre_trained,
                                                      unfreeze_weights=self.unfreeze_weights,
                                                      train_epochs=train_epochs,
                                                      learning_rate=learning_rate,
                                                      loss_function=self.loss_function,
                                                      predefined_datasets=None)

                self.train_label_table=clf.train_data_set.input_data
                self.valid_label_table=clf.valid_data_set.input_data
                self.test_label_table=clf.test_data_set.input_data
                self.datasets = {'train':self.train_label_table, 'valid':self.valid_label_table,'test':self.test_label_table}
                self.classifiers.append(clf)

            else:
                clf = Image_Classification(data_directory = self.data_directory,
                                                      name = None,
                                                      transformations=self.transformations,
                                                      custom_resize = self.custom_resize,
                                                      device=self.device,
                                                      optimizer=self.optimizer,
                                                      is_dicom=self.is_dicom,
                                                      label_from_table=self.label_from_table,
                                                      is_csv=self.is_csv,
                                                      table_source=self.table_source,
                                                      path_col = self.path_col,
                                                      label_col = self.label_col ,
                                                      balance_class = balance_class,
                                                      multi_label = self.multi_label,
                                                      mode=self.mode,
                                                      wl=self.wl,
                                                      normalize=normalize,
                                                      batch_size=batch_size,
                                                      test_percent = test_percent,
                                                      valid_percent = valid_percent,
                                                      model_arch=model_arch,
                                                      pre_trained=pre_trained,
                                                      unfreeze_weights=self.unfreeze_weights,
                                                      train_epochs=train_epochs,
                                                      learning_rate=learning_rate,
                                                      loss_function=self.loss_function,
                                                      predefined_datasets=self.datasets)
                self.classifiers.append(clf)

    def info(self):
      return self.scenarios_df

    def parameters(self):
        return self.variables_names

    def run(self):
      self.master_metrics = []
      self.trained_models = []
      for i in tqdm(self.classifiers, total=len(self.classifiers)):
        print ('Starting Training Classifier Number',self.classifiers.index(i))
        i.run()
        self.trained_models.append(i.trained_model)
        self.master_metrics.append(i.train_metrics)
        torch.cuda.empty_cache()

    def metrics(self, fig_size=(650,400)):
        return show_metrics(self.classifiers,  fig_size=fig_size)

    def roc(self, fig_size=(700,400)):
        self.auc_list = show_roc(self.classifiers, fig_size=fig_size)
        self.best_model_auc = max(self.auc_list)
        self.best_model_index = (self.auc_list.indxx(self.best_model_auc))
        self.best_classifier = self.classifiers.index(self.best_model_index)

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

        print ('best classifier')
