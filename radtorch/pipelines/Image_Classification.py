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
        self.custom_resize = custom_resize


        # Transformations
        self.transformations, self.input_resize = set_transformations(model_arch=self.model_arch, custom_resize=self.custom_resize, is_dicom=self.is_dicom, transformations=transformations)

        # Device
        self.device = set_device(device)

        if self.predefined_datasets:
            self.train_data_set, self.valid_data_set, self.test_data_set = load_predefined_datatables(
                                        data_directory=self.data_directory,
                                        is_csv=self.is_csv,
                                        is_dicom=self.is_dicom,
                                        predefined_datasets=self.predefined_datasets,
                                        path_col=self.path_col,
                                        label_col=self.label_col,
                                        mode=self.mode,
                                        wl=self.wl,
                                        transformations=self.transformations )
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

        # Display train breakdown by class
        info_dict['train_dataset'] = show_dataset_info(self.train_data_set)
        info_dict['train_dataset'].style.set_caption("train_dataset")
        # Display valid breakdown by class
        info_dict['valid_dataset'] = show_dataset_info(self.valid_data_set)
        info_dict['valid_dataset'].style.set_caption("valid_dataset")

        # Display test breakdown by class
        if self.test_percent > 0:
            info_dict['test_dataset'] = show_dataset_info(self.test_data_set)
            info_dict['test_dataset'].style.set_caption("test_dataset")



        if plot:
            plot_dataset_info(info_dict, plot_size= plot_size)
            # plot_pipline_dataset_info(info, test_percent = self.test_percent)
        else:

            display (show_dataset_info(self.train_data_set))
            display (show_dataset_info(self.valid_data_set))
            display (show_dataset_info(self.test_data_set))

    def sample(self, fig_size=(10,10), show_labels=True, show_file_name=False):
        '''
        Display sample of the training dataset.
        Inputs:
            num_of_images_per_row: _(int)_ number of images per column. (default=5)
            fig_size: _(tuple)_figure size. (default=(10,10))
            show_labels: _(boolean)_ show the image label idx. (default=True)
        '''
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

    def classes(self):
        return self.data_set.class_idx

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




if __name__ == "__main__":
    Image_Classification()
