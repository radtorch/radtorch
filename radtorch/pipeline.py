

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


from radtorch.modelsutils import create_model, create_loss_function, train_model, model_inference, model_dict, create_optimizer, supported_image_classification_losses
from radtorch.datautils import dataset_from_folder, dataset_from_table
from radtorch.visutils import show_dataset_info, show_dataloader_sample, show_metrics, show_confusion_matrix, show_roc, show_nn_roc




class Image_Classification():
    """

    The Image Classification pipeline simplifies the process of binary and multi-class image classification into a single line of code.
    Under the hood, the following happens:

    1. The pipeline creates a master dataset from the provided data directory and source of labels/classes either from [folder structre](https://pytorch.org/docs/stable/torchvision/datasets.html#datasetfolder) or pandas/csv table.

    2. Master dataset is subdivided into train, valid and test subsets using the percentages defined by user.

    3. The following transformations are applied on the dataset images:
        1. Resize to the default image size allowed by the model architecture.
        2. Window/Level adjustment according to values specified by user.
        3. Single channel grayscale DICOM images are converted into 3 channel grayscale images to fit into the model.

    3. Selected Model architecture, optimizer, and loss function are downloaded/created.

    4. Model is trained.

    5. Training metrics are saved as training progresses and can be displayed after training is done.

    6. Confusion Matrix and ROC (for binary classification) can be displayed as well (by default, the test subset is used to calculate the confusion matrix and the ROC)

    7. Trained model can be exported to outside file for future use.


####Parameters

!!! quote ""

    **data_directory:**

    - _(str)_ target data directory. ***(Required)***

    **is_dicom:**

    - _(boolean)_ True for DICOM images, False for regular images.(default=True)

    **label_from_table:**

    - _(boolean)_ True if labels are to extracted from table, False if labels are to be extracted from subfolders. (default=False)

    **is_csv:**

    - _(boolean)_ True for csv, False for pandas dataframe.

    **table_source:** _(str or pandas dataframe object)_ source for labelling data.This is path to csv file or name of pandas dataframe if pandas to be used. (default=None).

    **path_col:**

    - _(str)_  name of the column with the image path. (default='IMAGE_PATH')

    **label_col:**

    - _(str)_  name of the label/class column. (default='IMAGE_LABEL')

    **mode:** _(str)_  output mode for DICOM images only where RAW= Raw pixels, HU= Image converted to Hounsefield Units, WIN= 'window' image windowed to certain W and L, MWIN = 'multi-window' converts image to 3 windowed images of different W and L (specified in wl argument) stacked together. (default='RAW')

    **wl:** _(list)_ list of lists of combinations of window level and widths to be used with WIN and MWIN.In the form of : [[Level,Width], [Level,Width],...].  Only 3 combinations are allowed for MWIN (for now). (default=None)

    **transformations:** _(pytorch transforms list)_ pytroch transforms to be performed on the dataset. (default=Convert to tensor)

    **custom_resize:** _(int)_ by default, a radtorch pipeline will resize the input images into the default training model input image size as demosntrated in the table shown in radtorch home page. This default size can be changed here if needed.

    **batch_size:** _(int)_ batch size of the dataset (default=16)

    **test_percent:** _(float)_ percentage of dataset to use for testing. Float value between 0 and 1.0. (default=0.2)

    **valid_percent:** _(float)_ percentage of dataset to use for validation. Float value between 0 and 1.0. (default=0.2)

    **model_arch:** _(str)_ PyTorch neural network architecture (default='vgg16')

    **pre_trained:** _(boolean)_ Load the pretrained weights of the neural network. (default=True)

    **unfreeze_weights:** _(boolean)_ if True, all model weights will be retrained. (default=True)

    **train_epochs:** _(int)_ Number of training epochs. (default=20)

    **learning_rate:** _(str)_ training learning rate. (default = 0.0001)

    **loss_function:** _(str)_ training loss function. (default='CrossEntropyLoss')

    **optimizer:** _(str)_ Optimizer to be used during training. (default='Adam')

    **device:** _(str)_ device to be used for training. This can be adjusted to 'cpu' or 'cuda'. If nothing is selected, the pipeline automatically detects if cuda is available and trains on it.


####Methods

!!! quote ""


    **info**

    - Display Parameters of the Image Classification Pipeline.

    **dataset_info**

    - Display Dataset Information.

    **sample**

    - Display sample of the training dataset.

    - Arguments:
        - num_of_images_per_row: _(int)_ number of images per column. (default=5)
        - fig_size: _(tuple)_ figure size. (default=(10,10))
        - show_labels: _(boolean)_ show the image label idx. (default=True)

    **train**

    - Train the image classification pipeline.

    - Arguments:
        - verbose: _(boolean)_ Show display progress after each epoch. (default=True)

    **metrics**

    - Display the training metrics.

    **export_model**

    - Export the trained model into a target file.

    - Arguments:
        - output_path: _(str)_ path to output file. For example 'foler/folder/model.pth'


    **set_trained_model**

    - Loads a previously trained model into pipeline

    - Arguments:
        - model_path: _(str)_ path to target model
        - mode: _(str)_ either 'train' or 'infer'.'train' will load the model to be trained. 'infer' will load the model for inference.


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
    path_col = 'IMAGE_PATH',
    label_col = 'IMAGE_LABEL' ,
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
        self.optimizer = optimizer
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



        valid_size = int(self.valid_percent*len(self.data_set))
        test_size = int(self.test_percent*len(self.data_set))
        train_size = len(self.data_set) - (valid_size+test_size)

        self.train_data_set, self.valid_data_set, self.test_data_set = torch.utils.data.random_split(self.data_set, [train_size, valid_size, test_size])

        self.train_data_loader = torch.utils.data.DataLoader(
                                                    self.train_data_set,
                                                    batch_size=self.batch_size,
                                                    shuffle=True)

        self.valid_data_loader = torch.utils.data.DataLoader(
                                                    self.valid_data_set,
                                                    batch_size=self.batch_size,
                                                    shuffle=True)

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

        self.optimizer = create_optimizer(traning_model=self.train_model, optimizer_type=optimizer, learning_rate=self.learning_rate)


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
        print ('Test Dataset Size =', len(self.test_data_set))

    def dataset_info(self):
        '''
        Display Dataset Information.
        '''

        print (show_dataset_info(self.data_set))
        print ('Train Dataset Size ', len(self.train_data_set))
        print ('Valid Dataset Size ', len(self.valid_data_set))
        print ('Test Dataset Size ', len(self.test_data_set))

    def sample(self, num_of_images_per_row=5, fig_size=(10,10), show_labels=True):
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
            target_data_set = self.test_data_set
        else:
            target_data_set = target_data_set

        if target_classes == 'default':
            target_classes = self.data_set.classes
        else:
            target_classes = target_classes

        show_confusion_matrix(model=self.trained_model, target_data_set=target_data_set, target_classes=target_classes, figure_size=figure_size, cmap=cmap)


    def roc(self, target_data_set='default', auc=True, figure_size=(7,7)):
        '''
        Display Confusion Matrix
        Inputs:
            target_data_set: _(pytorch dataset object)_ dataset used for predictions to create the ROC. By default, the image classification pipeline uses the test dataset created to calculate the ROC.
            auc: _(boolen)_ Display area under curve. (default=True)
            figure_size: _(tuple)_figure size. (default=(7,7))
        '''

        if target_data_set=='default':
            target_data_set = self.test_data_set
        else:
            target_data_set = target_data_set

        show_nn_roc(model=self.trained_model, target_data_set=target_data_set, auc=auc, figure_size=figure_size)


class Feature_Extraction():
    """

    The feature extraction pipeline utilizes a pre-trained model to extract a set of features that can be used in another machine learning algorithms e.g. XGBoost. The trained model by default can one of the supported model architectures trained with default weights trained on the ImageNet dataset or a model that has been trained and exported using the image classification pipeline.

    The output is a pandas dataframe that has feature columns, label column and file path column.

    Under the hood, the pipeline removes the last FC layer of the pretrained models to output the features.

    The number of extracted features depends on the model architecture selected:

    <div align='center'>

    | Model Architecture | Default Input Image Size | Output Features |
    |--------------------|:------------------------:|:---------------:|
    | VGG16              |         244 x 244        |       4096      |
    | VGG19              |         244 x 244        |       4096      |
    | resnet50           |         244 x 244        |       2048      |
    | resnet152          |         244 x 244        |       2048      |
    | resnet101          |         244 x 244        |       2048      |
    | wide_resnet50_2    |         244 x 244        |       2048      |
    | wide_resnet101_2   |         244 x 244        |       2048      |
    | inception_v3       |         299 x 299        |       2048      |

    </div>


####Parameters

!!! quote ""

    **is_dicom:** _(boolean)_  True for DICOM images, False for regular images.(default=True)

    **label_from_table:** [boolean] True if labels are to extracted from table, False if labels are to be extracted from subfolders. (default=False)

    **is_csv:** _(boolean)_  True for csv, False for pandas dataframe.

    **table_source:** _(str or pandas dataframe object)_ source for labelling data. (default=None)
                This is path to csv file or name of pandas dataframe if pandas to be used.

    **path_col:** _(str)_ name of the column with the image path. (default='IMAGE_PATH')

    **label_col:** _(str)_ name of the label/class column. (default='IMAGE_LABEL')

    **mode:** _(str)_ output mode for DICOM images only.
              .Options:
                   RAW= Raw pixels,
                   HU= Image converted to Hounsefield Units,
                   WIN= 'window' image windowed to certain W and L,
                   MWIN = 'multi-window' converts image to 3 windowed images of different W and L (specified in wl argument) stacked together]. (default='RAW')

    **wl:** _(list)_ list of lists of combinations of window level and widths to be used with WIN and MWIN.
              In the form of : [[Level,Width], [Level,Width],...].
              Only 3 combinations are allowed for MWIN (for now).(default=None)

    **transformations:** _(pytorch transforms)_ pytroch transforms to be performed on the dataset. (default=Convert to tensor)

    **custom_resize:** _(int)_ by default, a radtorch pipeline will resize the input images into the default training model input image
    size as demosntrated in the table shown in radtorch home page. This default size can be changed here if needed.
    model_arch: [str] PyTorch neural network architecture (default='vgg16')

    **pre_trained:** _(boolean)_  Load the pretrained weights of the neural network. If False, the last layer is only retrained = Transfer Learning. (default=True)

    **unfreeze_weights:** _(boolean)_  if True, all model weights, not just final layer, will be retrained. (default=False)

    **device:** _(str)_ device to be used for training. This can be adjusted to 'cpu' or 'cuda'. If nothing is selected, the pipeline automatically detects if cuda is available and trains on it.


####Methods

!!! quote ""


    **info**

    - Displays Feature Extraction Pipeline Parameters.

    **dataset_info**

    - Display Dataset Information.

    **sample**

    - Display sample of the training dataset.

    **num_features**

    - Displays number of features to be extracted.

    **run**

    - Extracts features from dataset.

    - Arguments:
        - verbose: _(boolean)_ Show the feature table. (default=True)

    **export_features**

    - Exports the features to csv.

    - Arguments:
        - csv_path: _(str)_ Path to output csv file.


    **set_trained_model**

    - Loads a previously trained model into pipeline

    - Arguments:
        - model_path: _(str)_ path to target model
        - mode: _(str)_ either 'train' or 'infer'.'train' will load the model to be trained. 'infer' will load the model for inference.

    """

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
    unfreeze_weights=False,
    ):
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
                                                    batch_size=10,
                                                    shuffle=True)


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
        print (show_dataset_info(self.data_set))


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
        with torch.no_grad():
            self.model.eval()
            for input, label, img_path in tqdm(self.data_set, total=len(self.data_set)):
                input = input.to(self.device)
                input = input.unsqueeze(0)
                output = (self.model(input))[0].tolist()
                self.features.append(output)
                self.labels_idx.append(label)
                self.img_path_list.append(img_path)

        self.feature_names = ['f_'+str(i) for i in range(0,(model_dict[self.model_arch]['output_features']))]

        feature_df = pd.DataFrame(list(zip(self.img_path_list, self.labels_idx, self.features)), columns=['img_path','label_idx', 'features'])

        feature_df[self.feature_names] = pd.DataFrame(feature_df.features.values.tolist(), index= feature_df.index)

        print (' Features extracted successfully.')

        self.feature_df = feature_df

        if verbose:
            self.feature_df

        return self.feature_df


    def export_features(self,csv_path):
        try:
            self.feature_df.to_csv(csv_path)
            print ('Features exported to CSV successfully.')
        except:
            print ('Error! No features found. Please check again or re-run the extracion pipeline.')
            pass



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
