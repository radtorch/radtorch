# Copyright (C) 2020 RADTorch and Mohamed Elbanan, MD
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see https://www.gnu.org/licenses/

# Documentation update: 5/11/2020

from radtorch.settings import *
from radtorch.general import *
from radtorch.data import *
from radtorch.utils import *





class RADTorch_Dataset(Dataset):

    """

    Description
    -----------
    Core class for dataset. This is an extension of Pytorch dataset class with modifications.


    Parameters
    ------------
    - data_directory (string, required): path to target data directory/folder.

    - is_dicom (bollean, optional): True if images are DICOM. default=False.

    - table (string or pandas dataframe, optional): path to label table csv or name of pandas data table. default=None.

    - image_path_column (string, optional): name of column that has image path/image file name. default='IMAGE_PATH'.

    - image_label_column (string, optional): name of column that has image label. default='IMAGE_LABEL'.

    - is_path (boolean, optional): True if file_path column in table is file path. If False, this assumes that the column contains file names only and will append the data_directory to all files. default=True.

    - mode (string, optional): mode of handling pixel values from DICOM to numpy array. Option={'RAW': raw pixel values, 'HU': converts pixel values to HU using slope and intercept, 'WIN':Applies a certain window/level to HU converted DICOM image, 'MWIN': converts DICOM image to 3 channel HU numpy array with each channel adjusted to certain window/level. default='RAW'.

    - wl (tuple or list of tuples, optional): value of Window/Levelto be used. If mode is set to 'WIN' then wl takes the format (level, window). If mode is set to 'MWIN' then wl takes the format [(level1, window1), (level2, window2), (level3, window3)]. default=None.

    - sampling (float, optional): fraction of the whole dataset to be used. default=1.0.

    - transformations (list, optional): list of pytorch transformations to be applied to all datasets. By default, the images are resized, channels added up to 3 and greyscaled. default='default'.


    Returns
    -----------
    RADTorch dataset object.


    """

    def __init__(
                self,
                data_directory,
                transformations,
                table=None,
                is_dicom=False,
                mode='RAW',
                wl=None,
                image_path_column='IMAGE_PATH',
                image_label_column='IMAGE_LABEL',
                is_path=True,
                sampling=1.0,
                **kwargs):

        self.data_directory=data_directory
        self.transformations=transformations
        self.table=table
        self.is_dicom=is_dicom
        self.mode=mode
        self.wl=wl
        self.image_path_column=image_path_column
        self.image_label_column=image_label_column
        self.is_path=is_path
        self.sampling=sampling

        # Create Data Table
        if isinstance(self.table, pd.DataFrame): self.input_data=self.table
        elif isinstance(self.table, string): self.input_data=pd.read_csv(self.table)
        else: self.input_data=create_data_table(data_directory=self.data_directory, is_dicom=self.is_dicom, image_path_column=self.image_path_column, image_label_column=self.image_label_column)

        # Check if file path or file name and fix
        if self.is_path==False:
            files=[]
            for i, r in self.input_data.iterrows():
                files.append(self.data_directory+r[self.image_path_column])
            # self.table[self.image_path_column]=files
            self.input_data[self.image_path_column]=pd.Series(files, index=self.input_data.index)

        # Get list of files
        if self.is_dicom: self.dataset_files=[x for x in (self.input_data[self.image_path_column].tolist()) if x.endswith('.dcm')]
        else: self.dataset_files=[x for x in (self.input_data[self.image_path_column].tolist()) if x.endswith(IMG_EXTENSIONS)]

        # Get list of labels and create label dictionary
        self.classes= list(self.input_data[self.image_label_column].unique())
        self.class_to_idx=class_to_idx(self.classes)

        # Print Errors if classes or files = 0
        if len(self.dataset_files)==0: log ('Error! No data files found in directory:'+ self.data_directory)
        if len(self.classes)==0:log ('Error! No classes extracted from directory:'+ self.data_directory)

    def __getitem__(self, index):
        """
        Handles how to get an image of the dataset.
        """
        image_path=self.input_data.iloc[index][self.image_path_column]
        if self.is_dicom:
            image=dicom_to_narray(image_path, self.mode, self.wl)
            image=Image.fromarray(image)
        else:
            image=Image.open(image_path).convert('RGB')
        image=self.transformations(image)
        label=self.input_data.iloc[index][self.image_label_column]
        label_idx=[v for k, v in self.class_to_idx.items() if k == label][0]
        return image, label_idx, image_path

    def __len__(self):
        """
        Returns number of images in dataset.
        """
        return len(self.dataset_files)

    def info(self):
        """
        Returns information of the dataset.
        """
        return show_dataset_info(self)

    def classes(self):
        """
        returns list of classes in dataset.
        """
        return self.classes

    def class_to_idx(self):
        """
        returns mapping of classes to class id (dictionary).
        """
        return self.class_to_idx

    def parameters(self):
        """
        returns all the parameter names of the dataset.
        """
        return self.__dict__.keys()

    def balance(self, method='upsample'):
        """
        Retuns a balanced dataset. methods={'upsample', 'downsample'}
        """
        return balance_dataset(dataset=self, label_col=self.image_label_column, method=method)

    def mean_std(self):
        """
        calculates mean and standard deviation of dataset.
        """
        self.mean, self.std= calculate_mean_std(torch.utils.data.DataLoader(dataset=self))
        return tuple(self.mean.tolist()), tuple(self.std.tolist())

    def normalize(self, **kwargs):
        """
        Returns a normalized dataset with either mean/std of the dataset or a user specified mean/std in the form of ((mean, mean, mean), (std, std, std)).
        """
        if 'mean' in kwargs.keys() and 'std' in kwargs.keys():
            mean=kwargs['mean']
            std=kwargs['std']
        else:
            mean, std=self.mean_std()
        normalized_dataset=copy.deepcopy(self)
        normalized_dataset.transformations.transforms.append(transforms.Normalize(mean=mean, std=std))
        return normalized_dataset


class Data_Processor():

    """

    Description
    ------------
    Class Data Processor. The core class for data preparation before feature extraction and classification. This class performs dataset creation, data splitting, sampling, balancing, normalization and transformations.


    Parameters
    ------------

    - data_directory (string, required): path to target data directory/folder.

    - is_dicom (bollean, optional): True if images are DICOM. default=False.

    - table (string or pandas dataframe, optional): path to label table csv or name of pandas data table. default=None.

    - image_path_column (string, optional): name of column that has image path/image file name. default='IMAGE_PATH'.

    - image_label_column (string, optional): name of column that has image label. default='IMAGE_LABEL'.

    - is_path (boolean, optional): True if file_path column in table is file path. If False, this assumes that the column contains file names only and will append the data_directory to all files. default=False.

    - mode (string, optional): mode of handling pixel values from DICOM to numpy array. Option={'RAW': raw pixel values, 'HU': converts pixel values to HU using slope and intercept, 'WIN':Applies a certain window/level to HU converted DICOM image, 'MWIN': converts DICOM image to 3 channel HU numpy array with each channel adjusted to certain window/level. default='RAW'.

    - wl (tuple or list of tuples, optional): value of Window/Levelto be used. If mode is set to 'WIN' then wl takes the format (level, window). If mode is set to 'MWIN' then wl takes the format [(level1, window1), (level2, window2), (level3, window3)]. default=None.

    - balance_class (bollean, optional): True to perform oversampling in the train dataset to solve class imbalance. default=False.

    - balance_class_method (string, optional): methodology used to balance classes. Options={'upsample', 'downsample'}. default='upsample'.

    - normalize (bolean/False or Tuple, optional): Normalizes all datasets by a specified mean and standard deviation. Since most of the used CNN architectures assumes 3 channel input, this follows the following format ((mean, mean, mean), (std, std, std)). default=((0,0,0), (1,1,1)).

    - batch_size (integer, optional): Batch size for dataloader. defult=16.

    - num_workers (integer, optional): Number of CPU workers for dataloader. default=0.

    - sampling (float, optional): fraction of the whole dataset to be used. default=1.0.

    - test_percent (float, optional): percentage of data for testing.default=0.2.

    - valid_percent (float, optional): percentage of data for validation (ONLY with NN_Classifier) .default=0.2.

    - custom_resize (integer, optional): By default, the data processor resizes the image in dataset into the size expected bu the different CNN architectures. To override this and use a custom resize, set this to desired value. default=False.

    - model_arch (string, required): CNN model architecture that this data will be used for. Used to resize images as detailed above. default='alexnet' .

    - type (string, required): type of classifier that will be used. please refer to classifier object type. default='nn_classifier'.

    - device (string, optional): device to be used for training. Options{'auto': automatic detection of device type, 'cpu': cpu, 'cuda': gpu}. default='auto'.

    - transformations (list, optional): list of pytorch transformations to be applied to all datasets. By default, the images are resized, channels added up to 3 and greyscaled. default='default'.

    - extra_transformations (list, optional): list of pytorch transformations to be extra added to train dataset specifically. default=None.


    """

    def __init__(
                self,
                data_directory,
                is_dicom=False,
                table=None,
                image_path_column='IMAGE_PATH',
                image_label_column='IMAGE_LABEL',
                is_path=True,
                mode='RAW',
                wl=None,
                balance_class=False,
                balance_class_method='upsample',
                normalize=((0,0,0), (1,1,1)),
                batch_size=16,
                num_workers=0,
                sampling=1.0,
                custom_resize=False,
                model_arch='alexnet',
                type='nn_classifier',
                transformations='default',
                extra_transformations=None,
                test_percent=0.2,
                valid_percent=0.2,
                device='auto',
                **kwargs):

        self.data_directory=data_directory
        self.is_dicom=is_dicom
        self.table=table
        self.image_path_column=image_path_column
        self.image_label_column=image_label_column
        self.is_path=is_path
        self.mode=mode
        self.wl=wl
        self.balance_class=balance_class
        self.balance_class_method=balance_class_method
        self.normalize=normalize
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.sampling=sampling
        self.custom_resize=custom_resize
        self.model_arch=model_arch
        self.type=type
        self.transformations=transformations
        self.extra_transformations=extra_transformations
        self.device=device
        self.test_percent=test_percent
        self.valid_percent=valid_percent


        if self.device=='auto': self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create Initial Master Table
        if isinstance(self.table, str):
            if self.table!='':
                self.table=pd.read_csv(self.table)
        elif isinstance(self.table, pd.DataFrame):
            self.table=self.table
        else: self.table=create_data_table(directory=self.data_directory, is_dicom=self.is_dicom, image_path_column=self.image_path_column, image_label_column=self.image_label_column)


        # Sample from dataset if necessary
        if isinstance (self.sampling, float):
            if self.sampling > 1.0 :
                log('Error! Sampling cannot be more than 1.0.')
                pass
            elif self.sampling == 0:
                log ('Error! Sampling canot be Zero.')
                pass
            else:
                self.table=self.table.sample(frac=self.sampling, random_state=100)
        else:
            log ('Error! Sampling is not float')
            pass


        # Split into test, valid and train
        self.temp_table, self.test_table=train_test_split(self.table, test_size=self.test_percent, random_state=100, shuffle=True)
        self.train_table, self.valid_table=train_test_split(self.temp_table, test_size=(len(self.table)*self.valid_percent/len(self.temp_table)), random_state=100, shuffle=True)

        # Define Transformations
        # 1- Custom Resize Adjustement
        if self.custom_resize in [False, '', 0, None]: self.resize=model_dict[self.model_arch]['input_size']
        elif isinstance(self.custom_resize, int): self.resize=self.custom_resize
        else: log ('Image Custom Resize not allowed. Please recheck.')

        # 2- Image conversion from DICOM
        if self.transformations=='default':
            if self.is_dicom:
                self.transformations=transforms.Compose([
                        transforms.Resize((self.resize, self.resize)),
                        transforms.transforms.Grayscale(3),
                        transforms.ToTensor()])
            else:
                self.transformations=transforms.Compose([
                    transforms.Resize((self.resize, self.resize)),
                    transforms.ToTensor()])


        # 3- Normalize Training Dataset
        self.train_transformations=copy.deepcopy(self.transformations)
        if self.extra_transformations != None :
            for i in self.extra_transformations:
                self.train_transformations.transforms.insert(1, i)
        if isinstance (self.normalize, tuple):
            mean, std=self.normalize
            self.train_transformations.transforms.append(transforms.Normalize(mean=mean, std=std))
        elif self.normalize!=False:
            log('Error! Selected mean and standard deviation are not allowed.')
            pass


        self.master_dataset=RADTorch_Dataset(
                                            data_directory=self.data_directory,
                                            table=self.table,
                                            is_dicom=self.is_dicom,
                                            mode=self.mode,
                                            wl=self.wl,
                                            image_path_column=self.image_path_column,
                                            image_label_column=self.image_label_column,
                                            is_path=self.is_path,
                                            sampling=1.0,
                                            transformations=self.transformations)

        self.num_output_classes=len(self.master_dataset.classes)

        if self.type=='nn_classifier':
            self.train_dataset=RADTorch_Dataset(
                                                data_directory=self.data_directory,
                                                table=self.train_table,
                                                is_dicom=self.is_dicom,
                                                mode=self.mode,
                                                wl=self.wl,
                                                image_path_column=self.image_path_column,
                                                image_label_column=self.image_label_column,
                                                is_path=self.is_path,
                                                sampling=1.0,
                                                transformations=self.train_transformations)
            if self.balance_class:
                self.train_dataset=self.train_dataset.balance(method=self.balance_class_method)
            self.valid_dataset=RADTorch_Dataset(
                                                data_directory=self.data_directory,
                                                table=self.valid_table,
                                                is_dicom=self.is_dicom,
                                                mode=self.mode,
                                                wl=self.wl,
                                                image_path_column=self.image_path_column,
                                                image_label_column=self.image_label_column,
                                                is_path=self.is_path,
                                                sampling=1.0,
                                                transformations=self.transformations)
            self.valid_dataloader=torch.utils.data.DataLoader(dataset=self.valid_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        else:
            self.train_dataset=RADTorch_Dataset(
                                                data_directory=self.data_directory,
                                                table=self.temp_table,
                                                is_dicom=self.is_dicom,
                                                mode=self.mode,
                                                wl=self.wl,
                                                image_path_column=self.image_path_column,
                                                image_label_column=self.image_label_column,
                                                is_path=self.is_path,
                                                sampling=1.0,
                                                transformations=self.train_transformations)
            if self.balance_class:
                self.train_dataset=self.train_dataset.balance(method=self.balance_class_method)

        self.test_dataset=RADTorch_Dataset(
                                            data_directory=self.data_directory,
                                            table=self.test_table,
                                            is_dicom=self.is_dicom,
                                            mode=self.mode,
                                            wl=self.wl,
                                            image_path_column=self.image_path_column,
                                            image_label_column=self.image_label_column,
                                            is_path=self.is_path,
                                            sampling=1.0,
                                            transformations=self.transformations)

        self.master_dataloader=torch.utils.data.DataLoader(dataset=self.master_dataset,batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.train_dataloader=torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_dataloader=torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def classes(self):
        """
        Returns dictionary of classes/class_idx in data.
        """
        return self.master_dataset.class_to_idx

    def info(self):
        """
        Returns full information of the data processor object.
        """
        info=pd.DataFrame.from_dict(({key:str(value) for key, value in self.__dict__.items()}).items())
        info.columns=['Property', 'Value']
        info=info.append({'Property':'master_dataset_size', 'Value':len(self.master_dataset)}, ignore_index=True)
        for i in ['train_dataset', 'valid_dataset','test_dataset']:
            if i in self.__dict__.keys():
                info.append({'Property':i+'_size', 'Value':len(self.__dict__[i])}, ignore_index=True)
        return info

    def dataset_info(self, plot=True, figure_size=(500,300)):
        """
        Displays information of the data and class breakdown.

        Parameters
        -----------
        plot (boolean, optional): True to display data as graph. False to display in table format. default=True
        figure_size (tuple, optional): Tuple of width and lenght of figure plotted. default=(500,300)
        """

        info_dict={}
        info_dict['dataset']=show_dataset_info(self.master_dataset)
        info_dict['dataset'].style.set_caption('Overall Dataset')
        if 'type' in self.__dict__.keys():
            for i in ['train_dataset','test_dataset']:
                if i in self.__dict__.keys():
                    info_dict[i]= show_dataset_info(self.__dict__[i])
                    info_dict[i].style.set_caption(i)
            if self.type=='nn_classifier':
                if 'valid_dataset' in self.__dict__.keys():
                    info_dict['valid_dataset']= show_dataset_info(self.__dict__['valid_dataset'])
                    info_dict[i].style.set_caption('valid_dataset')

        if plot:
            plot_dataset_info(info_dict, plot_size= figure_size)
        else:
            for k, v in info_dict.items():
                display(v)

    def sample(self, figure_size=(10,10), show_labels=True, show_file_name=False):
        """
        Displays a sample from the training dataset. Number of images displayed is the same as batch size.

        Parameters
        ----------
        figure_size (tuple, optional): Tuple of width and lenght of figure plotted. default=(10,10)
        show_label (boolean, optional): show labels above images. default=True
        show_file_names (boolean, optional): show file path above image. default=False


        """
        show_dataloader_sample(self.train_dataloader, figure_size=figure_size, show_labels=show_labels, show_file_name=show_file_name)

    def check_leak(self, show_file=False):
        """
        Checks possible overlap between train and test dataset files.

        Parameters
        ----------
        show_file (boolean, optional): display table of leaked/common files between train and test. default=False.

        """
        train_file_list=self.train_dataset.input_data[self.image_path_column]
        test_file_list=self.test_dataset.input_data[self.image_path_column]
        leak_files=[]
        for i in train_file_list:
            if i in test_file_list:
                leak_files.append(i)
        log('Data Leak Check: '+str(len(train_file_list))+' train files checked. '+str(len(leak_files))+' common files were found in train and test datasets.')
        if show_file:
            return pd.DataFrame(leak_files, columns='leaked_files')

    def export(self, output_path):
        """
        Exports the Dtaprocessor object for future use.

        Parameters
        ----------
        output_path (string, required): output file path.

        """
        try:
            outfile=open(output_path,'wb')
            pickle.dump(self,outfile)
            outfile.close()
            log('Data Processor exported successfully.')
        except:
            raise TypeError('Error! Data Processor could not be exported.')


class Feature_Extractor():

    """

    Description
    -----------
    Creates a feature extractor neural network using one of the famous CNN architectures and the data provided as dataloader from Data_Processor.

    Parameters
    ----------

    - model_arch (string, required): CNN architecture to be utilized. To see list of supported architectures see settings.

    - pre_trained (boolean, optional): Initialize with ImageNet pretrained weights or not. default=True.

    - unfreeze (boolean, required): Unfreeze all layers of network for future retraining. default=False.

    - dataloader (pytorch dataloader object, required): the dataloader that will be used to supply data for feature extraction.

    - device (string, optional): device to be used for training. Options{'auto': automatic detection of device type, 'cpu': cpu, 'cuda': gpu}. default='auto'.

    Retruns
    ---------
    Pandas dataframe with extracted features.

    """


    def __init__(
                self,
                model_arch,
                dataloader,
                pre_trained=True,
                unfreeze=False,
                device='auto',
                **kwargs):

        self.model_arch=model_arch
        self.dataloader=dataloader
        self.pre_trained=pre_trained
        self.unfreeze=unfreeze
        self.device=device

        for k,v in kwargs.items():
            setattr(self,k,v)
        if self.model_arch not in supported_models:
            log('Error! Provided model architecture is not yet suported. Please use radtorch.settings.supported_models to see full list of supported models.')
            pass
        elif self.model_arch=='vgg11': self.model=torchvision.models.vgg11(pretrained=self.pre_trained)
        elif self.model_arch=='vgg13':  self.model=torchvision.models.vgg13(pretrained=self.pre_trained)
        elif self.model_arch=='vgg16':  self.model=torchvision.models.vgg16(pretrained=self.pre_trained)
        elif self.model_arch=='vgg19':  self.model=torchvision.models.vgg19(pretrained=self.pre_trained)
        elif self.model_arch=='vgg11_bn': self.model=torchvision.models.vgg11_bn(pretrained=self.pre_trained)
        elif self.model_arch=='vgg13_bn': self.model=torchvision.models.vgg13_bn(pretrained=self.pre_trained)
        elif self.model_arch=='vgg16_bn': self.model=torchvision.models.vgg16_bn(pretrained=self.pre_trained)
        elif self.model_arch=='vgg19_bn': self.model=torchvision.models.vgg19_bn(pretrained=self.pre_trained)
        elif self.model_arch=='resnet18': self.model=torchvision.models.resnet18(pretrained=self.pre_trained)
        elif self.model_arch=='resnet34': self.model=torchvision.models.resnet34(pretrained=self.pre_trained)
        elif self.model_arch=='resnet50': self.model=torchvision.models.resnet50(pretrained=self.pre_trained)
        elif self.model_arch=='resnet101': self.model=torchvision.models.resnet101(pretrained=self.pre_trained)
        elif self.model_arch=='resnet152': self.model=torchvision.models.resnet152(pretrained=self.pre_trained)
        elif self.model_arch=='wide_resnet50_2': self.model=torchvision.models.wide_resnet50_2(pretrained=self.pre_trained)
        elif self.model_arch=='wide_resnet101_2': self.model=torchvision.models.wide_resnet101_2(pretrained=self.pre_trained)
        elif self.model_arch=='alexnet': self.model=torchvision.models.alexnet(pretrained=self.pre_trained)

        if 'alexnet' in self.model_arch or 'vgg' in self.model_arch:
            self.model.classifier[6]=torch.nn.Identity()

        elif 'resnet' in self.model_arch:
            self.model.fc=torch.nn.Identity()

        if self.unfreeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def num_features(self):
        """
        Returns the number of features to be extracted.
        """
        return model_dict[self.model_arch]['output_features']

    def run(self, verbose=False):
        """
        Runs the feature exraction process

        Returns
        --------
        tuple of feature_table (dataframe which contains all features, labels and image file path), features (dataframe which contains features only), feature_names(list of feature names)

        """

        if 'balance_class' in self.__dict__.keys() and 'normalize' in self.__dict__.keys():
            log('Running Feature Extraction using '+str(self.model_arch)+' architecture with balance_class = '+str(self.balance_class)+' and normalize = '+str(self.normalize)+".")
        else:
            log('Running Feature Extraction using '+str(self.model_arch)+' architecture')
        self.features=[]
        self.labels_idx=[]
        self.img_path_list=[]
        self.model=self.model.to(self.device)
        for i, (imgs, labels, paths) in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
            self.labels_idx=self.labels_idx+labels.tolist()
            self.img_path_list=self.img_path_list+list(paths)
            with torch.no_grad():
                self.model.eval()
                imgs=imgs.to(self.device)
                output=(self.model(imgs)).tolist()
                self.features=self.features+(output)
        self.feature_names=['f_'+str(i) for i in range(0,self.num_features())]
        feature_table=pd.DataFrame(list(zip(self.img_path_list, self.labels_idx, self.features)), columns=['IMAGE_PATH','IMAGE_LABEL', 'FEATURES'])
        feature_table[self.feature_names]=pd.DataFrame(feature_table.FEATURES.values.tolist(), index= feature_table.index)
        feature_table=feature_table.drop(['FEATURES'], axis=1)
        log('Features extracted successfully.')
        self.feature_table=feature_table
        self.features=self.feature_table[self.feature_names]
        return self.feature_table, self.features, self.feature_names
        if verbose:
            print (self.feature_table)

    def export_features(self,csv_path):

        """
        Exports extracted features into csv file.

        Parameters
        ----------
        csv_path (string, required): path to csv output.

        """
        try:
            self.feature_table.to_csv(csv_path, index=False)
            log('Features exported to CSV successfully.')
        except:
            log('Error! No features found. Please check again or re-run the extracion pipeline.')
            pass

    def plot_extracted_features(self, num_features=100, num_images=100,image_path_column='IMAGE_PATH', image_label_column='IMAGE_LABEL'):
        """
        Plots Extracted Features in Heatmap

        Parameters
        -----------

        - num_features (integer, optional): number of features to display. default=100

        - num_images (integer, optional): number of images to display features for. default=100

        - image_path_column (string, required): name of column that has image names/path. default='IMAGE_PATH'

        - image_label_column (string, required): name of column that has image labels. default='IMAGE_LABEL'

        """

        return plot_features(feature_table=self.feature_table, feature_names=self.feature_names, num_features=num_features, num_images=num_images,image_path_col=image_path_column, image_label_col=image_label_column)

    def export(self, output_path):
        """
        Exports the Feature Extractor object for future use.

        Parameters
        ----------
        output_path (string, required): output file path.

        """
        try:
            outfile=open(output_path,'wb')
            pickle.dump(self,outfile)
            outfile.close()
            log('Feature Extractor exported successfully.')
        except:
            raise TypeError('Error! Feature Extractor could not be exported.')


class Classifier(object):

    """

    Description
    -----------
    Image Classification Class. Performs Binary/Multiclass classification using features extracted via Feature Extractor or Supplied by user.


    Parameters
    -----------

    - extracted_feature_dictionary (dictionary, required): Dictionary of features/labels datasets to be used for classification. This follows the following format :
    {
        'train':
                {'features':dataframe, 'feature_names':list, 'labels': list}},
        'test':
                {'features':dataframe, 'feature_names':list, 'labels': list}},
    }

    - feature_table (string, optional): path to csv table with user selected image paths, labels and features. default=None.

    - image_label_column (string, required if using feature_table): name of the column with images labels.default=None.

    - image_path_column (string, requried if using feature_table): name of column with images paths.default=None.

    - test_percent (float, required if using feature_table): percentage of data for testing.default=None.

    - type (string, required): type of classifier. For complete list refer to settings. default='logistic_regression'.

    - interaction_terms (boolean, optional): create interaction terms between different features and add them as new features to feature table. default=False.

    - cv (boolean, required): True for cross validation. default=True.

    - stratified (boolean, required): True for stratified cross validation. default=True.

    - num_splits (integer, required): Number of K-fold cross validation splits. default=5.

    - parameters (dictionary, optional): optional parameters passed to the classifier. Please refer to sci-kit learn documentaion.

    """

    def __init__(self,
                extracted_feature_dictionary,
                feature_table=None,
                image_label_column=None,
                image_path_column=None,
                test_percent=None,
                type='logistic_regression',
                interaction_terms=False,
                cv=True,
                stratified=True,
                num_splits=5,
                parameters={},
                **kwargs):


        self.extracted_feature_dictionary=extracted_feature_dictionary
        self.feature_table=feature_table
        self.image_label_column=image_label_column
        self.image_path_column=image_path_column
        self.test_percent=test_percent
        self.type=type
        self.interaction_terms=interaction_terms
        self.cv=cv
        self.stratified=stratified
        self.num_splits=num_splits
        self.parameters=parameters


        # Load extracted feature dictionary
        if 'extracted_feature_dictionary' in self.__dict__.keys():
            self.feature_names=self.extracted_feature_dictionary['train']['features_names']
            self.train_features=self.extracted_feature_dictionary['train']['features']
            self.train_labels=np.array(self.extracted_feature_dictionary['train']['labels'])
            self.test_features=self.extracted_feature_dictionary['test']['features']
            self.test_labels=np.array(self.extracted_feature_dictionary['test']['labels'])


        # Or Load user specified features
        else:
            if self.feature_table !=None:
                if isinstance(self.feature_table, str):
                    try:
                        self.feature_table=pd.read_csv(self.feature_table)
                    except:
                        log('Loading feature table failed. Please check the location of the feature table.')
                        pass
            self.feature_names=[x for x in self.feature_table.columns if x not in [self.image_label_column,self.image_path_column]]
            self.labels=self.feature_table[self.image_label_column]
            self.features=self.feature_table[self.feature_names]
            self.train_features,  self.test_features, self.train_labels, self.test_labels=train_test_split(self.features, self.labels, test_size=self.test_percent, random_state=100)

        # Interaction Terms
        if self.interaction_terms:
            log('Creating Interaction Terms for Train Dataset.')
            self.train_features=self.create_interaction_terms(self.train_features)
            log('Creating Interaction Terms for Test Dataset.')
            self.test_features=self.create_interaction_terms(self.test_features)
            log('Interaction Terms Created Successfully.')

        # Create Classifier object
        self.classifier=self.create_classifier(**self.parameters)
        self.classifier_type=self.classifier.__class__.__name__

    def create_classifier(self, **kw):

        """
        Creates Classifier Object
        """

        if self.type not in SUPPORTED_CLASSIFIER:
          log('Error! Classifier type not supported. Please check again.')
          pass
        elif self.type=='linear_regression':
          classifier=LinearRegression(n_jobs=-1, **kw)
        elif self.type=='logistic_regression':
          classifier=LogisticRegression(max_iter=10000,n_jobs=-1, **kw)
        elif self.type=='ridge':
          classifier=RidgeClassifier(max_iter=10000, **kw)
        elif self.type=='sgd':
          classifier=SGDClassifier(**kw)
        elif self.type=='knn':
          classifier=KNeighborsClassifier(n_jobs=-1,**kw)
        elif self.type=='decision_trees':
          classifier=tree.DecisionTreeClassifier(**kw)
        elif self.type=='random_forests':
          classifier=RandomForestClassifier(**kw)
        elif self.type=='gradient_boost':
          classifier=GradientBoostingClassifier(**kw)
        elif self.type=='adaboost':
          classifier=AdaBoostClassifier(**kw)
        elif self.type=='xgboost':
          classifier=XGBClassifier(**kw)
        return classifier

    def info(self):

        """
        Returns table of different classifier parameters/properties.
        """

        info=pd.DataFrame.from_dict(({key:str(value) for key, value in self.__dict__.items()}).items())
        info.columns=['Property', 'Value']
        return info

    def run(self):

        """
        Runs Image Classifier.
        """

        self.scores=[]
        self.train_metrics=[]
        if self.cv:
          if self.stratified:
            kf=StratifiedKFold(n_splits=self.num_splits, shuffle=True, random_state=100)
            log('Training '+str(self.classifier_type)+ ' with '+str(self.num_splits)+' split stratified cross validation.')
          else:
            kf=KFold(n_splits=self.num_splits, shuffle=True, random_state=100)
            log('Training '+str(self.classifier_type)+ ' classifier with '+str(self.num_splits)+' splits cross validation.')
          split_id=0
          for train, test in tqdm(kf.split(self.train_features, self.train_labels), total=self.num_splits):
            self.classifier.fit(self.train_features.iloc[train], self.train_labels[train])
            split_score=self.classifier.score(self.train_features.iloc[test], self.train_labels[test])
            self.scores.append(split_score)
            log('Split '+str(split_id)+' Accuracy = ' +str(split_score))
            self.train_metrics.append([[0],[0],[split_score],[0]])
            split_id+=1
        else:
          log('Training '+str(self.type)+' classifier without cross validation.')
          self.classifier.fit(self.train_features, self.train_labels)
          score=self.classifier.score(self.test_features, self.test_labels)
          self.scores.append(score)
          self.train_metrics.append([[0],[0],[score],[0]])
        self.scores = np.asarray(self.scores )
        self.classes=self.classifier.classes_.tolist()
        log(str(self.classifier_type)+ ' model training finished successfully.')
        log(str(self.classifier_type)+ ' overall training accuracy: %0.2f (+/- %0.2f)' % ( self.scores .mean(),  self.scores .std() * 2))
        self.train_metrics = pd.DataFrame(data=self.train_metrics, columns = ['Train_Loss', 'Valid_Loss', 'Train_Accuracy', 'Valid_Accuracy'])
        return self.classifier, self.train_metrics

    def average_cv_accuracy(self):

        """
        Returns average cross validation accuracy.
        """

        if self.cv:
          return self.scores.mean()
        else:
          log('Error! Training was done without cross validation. Please use test_accuracy() instead.')

    def test_accuracy(self) :

        """
        Returns accuracy of trained classifier on test dataset.
        """

        acc= self.classifier.score(self.test_features, self.test_labels)
        return acc

    def confusion_matrix(self,title='Confusion Matrix',cmap=None,normalize=False,figure_size=(8,6)):

        """
        Displays confusion matrix using trained classifier and test dataset.

        Parameters
        ----------

        - title (string, optional): name to be displayed over confusion matrix.

        - cmap (string, optional): colormap of the displayed confusion matrix. This follows matplot color palletes. default=None.

        - normalize (boolean, optional): normalize values. default=False.

        - figure_size (tuple, optional): size of the figure as width, height. default=(8,6)

        """

        pred_labels=self.classifier.predict(self.test_features)
        true_labels=self.test_labels
        cm = metrics.confusion_matrix(true_labels, pred_labels)
        show_confusion_matrix(cm=cm,
                              target_names=self.classes,
                              title=title,
                              cmap=cmap,
                              normalize=normalize,
                              figure_size=figure_size
                              )

    def roc(self, **kw):

        """
        Display ROC and AUC of trained classifier and test dataset.

        """

        show_roc([self], **kw)

    def predict(self, input_image_path, all_predictions=False, **kw):

        """

        Description
        -----------
        Returns label prediction of a target image using a trained classifier. This works as part of pipeline only for now.


        Parameters
        ----------

        - input_image_path (string, required): path of target image.

        - all_predictions (boolean, optional): return a table of all predictions for all possible labels.


        """

        classifier=self.classifier

        transformations=self.data_processor.transformations

        model=self.feature_extractor.model

        if input_image_path.endswith('dcm'):
            target_img=dicom_to_pil(input_image_path)
        else:
            target_img=Image.open(input_image_path).convert('RGB')

        target_img_tensor=transformations(target_img)
        target_img_tensor=target_img_tensor.unsqueeze(0)

        with torch.no_grad():
            model.to('cpu')
            target_img_tensor.to('cpu')
            model.eval()
            out=model(target_img_tensor)
        image_features=pd.DataFrame(out, columns=self.feature_names)

        class_to_idx = self.data_processor.classes()

        if all_predictions:
            try:
                A = self.data_processor.classes().keys()
                B = self.data_processor.classes().values()
                C = self.classifier.predict_proba(image_features)[0]
                C = [("%.4f" % x) for x in C]
                return pd.DataFrame(list(zip(A, B, C)), columns=['LABEL', 'LAEBL_IDX', 'PREDICTION_ACCURACY'])
            except:
                log('All predictions could not be generated. Please set all_predictions to False.')
                pass
        else:
            prediction=self.classifier.predict(image_features)

            return (prediction[0], [k for k,v in class_to_idx.items() if v==prediction][0])

    def export(self, output_path):

        """
        Exports the Classifier object for future use.

        Parameters
        ----------
        output_path (string, required): output file path.

        """
        try:
          outfile=open(output_path,'wb')
          pickle.dump(self,outfile)
          outfile.close()
          log('Classifier exported successfully.')
        except:
          raise TypeError('Error! Classifier could not be exported.')

    def export_trained_classifier(self, output_path):
        """
        Exports the trained classifier for future use.

        Parameters
        ----------
        output_path (string, required): output file path.

        """
        try:
          outfile=open(output_path,'wb')
          pickle.dump(self.classifier,outfile)
          outfile.close()
          log('Trained Classifier exported successfully.')
        except:
          raise TypeError('Error! Trained Classifier could not be exported.')

    # NEEDS TESTING
    def misclassified(self, num_of_images=4, figure_size=(5,5), table=False, **kw): # NEEDS CHECK FILE PATH !!!!!
      pred_labels=(self.classifier.predict(self.test_features)).tolist()
      true_labels=self.test_labels.tolist()
      accuracy_list=[0.0]*len(true_labels)

      y = copy.deepcopy(self.test_features)
      paths=[]
      for i in y.index.tolist():paths.append(self.test_feature_extractor.feature_table.iloc[i]['IMAGE_PATH'])  # <<<<< this line was changed .. check. / Accuracy not showing correctly !!

      misclassified_dict=misclassified(true_labels_list=true_labels, predicted_labels_list=pred_labels, accuracy_list=accuracy_list, img_path_list=paths)
      show_misclassified(misclassified_dictionary=misclassified_dict, transforms=self.data_processor.transformations, class_to_idx_dict=self.data_processor.classes(), is_dicom = self.is_dicom, num_of_images = num_of_images, figure_size =figure_size)
      misclassified_table = pd.DataFrame(misclassified_dict.values())
      if table:
          return misclassified_table

    # NEEDS TESTING
    def coef(self, figure_size=(50,10), plot=False):#BETA
      coeffs = pd.DataFrame(dict(zip(self.feature_names, self.classifier.coef_.tolist())), index=[0])
      if plot:
          coeffs.T.plot.bar(legend=None, figsize=figure_size);
      else:
          return coeffs

    # NEEDS TESTING
    def create_interaction_terms(self, table):#BETA
        self.interaction_features=table.copy(deep=True)
        int_feature_names = self.interaction_features.columns
        m=len(int_feature_names)
        for i in tqdm(range(m)):
            feature_i_name = int_feature_names[i]
            feature_i_data = self.interaction_features[feature_i_name]
            for j in range(i+1, m):
                feature_j_name = int_feature_names[j]
                feature_j_data = self.interaction_features[feature_j_name]
                feature_i_j_name = feature_i_name+'_x_'+feature_j_name
                self.interaction_features[feature_i_j_name] = feature_i_data*feature_j_data
        return self.interaction_features


class NN_Classifier():

    """
    Description
    ------------
    Neural Network Classifier. This serves as extension of pytorch neural network modules e.g. VGG16, for fine tuning or transfer learning.


    Parameters
    ----------

    - data_processor (radtorch.core.data_processor, required): data processor object from radtorch.core.Data_Processor.

    - feature_extractor (radtorch.core.feature_extractor, required): feature_extractor object from radtorch.core.Feature_Extractor.

    - unfreeze (boolean, optional): True to unfreeze the weights of all layers in the neural network model for model finetuning. False to just use unfreezed final layers for transfer learning. default=False.

    - learning_rate (float, required): Learning rate. default=0.0001.

    - epochs (integer, required): training epochs. default=10.

    - optimizer (string, required): neural network optimizer type. Please see radtorch.settings for list of approved optimizers. default='Adam'.

    - optimizer_parameters (dictionary, optional): optional extra parameters for optimizer as per pytorch documentation.

    - loss_function (string, required): neural network loss function. Please see radtorch.settings for list of approved loss functions. default='CrossEntropyLoss'.

    - loss_function_parameters (dictionary, optional): optional extra parameters for loss function as per pytorch documentation.

    - lr_scheduler (string, optional): learning rate scheduler - upcoming soon.

    - batch_size (integer, required): batch size. default=16

    - custom_nn_classifier (pytorch model, optional): Option to use a custom made neural network classifier that will be added after feature extracted layers. default=None.

    - device (string, optional): device to be used for training. Options{'auto': automatic detection of device type, 'cpu': cpu, 'cuda': gpu}. default='auto'.

    -

    """

    def __init__(self,
                feature_extractor,
                data_processor,
                unfreeze=False,
                learning_rate=0.0001,
                epochs=10,
                optimizer='Adam',
                loss_function='CrossEntropyLoss',
                lr_scheduler=None,
                batch_size=16,
                device='auto',
                custom_nn_classifier=None,
                loss_function_parameters={},
                optimizer_parameters={},
                **kwargs):

        self.feature_extractor=feature_extractor
        self.data_processor=data_processor
        self.unfreeze=unfreeze
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.optimizer=optimizer
        self.loss_function=loss_function
        self.lr_scheduler=lr_scheduler
        self.batch_size=batch_size
        self.device=device
        self.custom_nn_classifier=custom_nn_classifier
        self.loss_function_parameters=loss_function_parameters
        self.optimizer_parameters=optimizer_parameters

        if self.device=='auto': self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.feature_extractor== None or self.data_processor== None:
            log('Error! No  Data Processor and/or Feature Selector was supplied. Please Check.')
            pass

        # DATA
        self.output_classes=self.data_processor.num_output_classes
        self.train_dataset=self.data_processor.train_dataset
        self.train_dataloader=self.data_processor.train_dataloader
        self.valid_dataset=self.data_processor.valid_dataset
        self.valid_dataloader=self.data_processor.valid_dataloader
        if self.data_processor.test_percent>0:
            self.test_dataset=self.data_processor.test_dataset
            self.test_dataloader=self.data_processor.test_dataloader
        self.transformations=self.data_processor.transformations


        # MODEL
        self.model=copy.deepcopy(self.feature_extractor.model)
        self.model_arch=self.feature_extractor.model_arch
        self.in_features=model_dict[self.model_arch]['output_features']

        if self.custom_nn_classifier !=None:
            if 'vgg' in self.model_arch or 'alexnet' in self.model_arch: self.model.classifier=self.custom_nn_classifier
            elif 'resnet' in self.model_arch: self.model.fc=self.custom_nn_classifier

        else:
            if 'vgg' in self.model_arch:
                self.model.classifier=torch.nn.Sequential(
                                torch.nn.Dropout(p=0.5),
                                torch.nn.Linear(in_features=25088, out_features=4096, bias=True),
                                torch.nn.ReLU(inplace=True),
                                torch.nn.Dropout(p=0.5),
                                torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
                                torch.nn.ReLU(inplace=True),
                                torch.nn.Linear(in_features=4096, out_features=self.output_classes, bias=True),
                                torch.nn.LogSoftmax(dim=1))

            elif 'alexnet' in self.model_arch:
                self.model.classifier=torch.nn.Sequential(
                                torch.nn.Dropout(p=0.5),
                                torch.nn.Linear(in_features=9216, out_features=4096, bias=True),
                                torch.nn.ReLU(inplace=True),
                                torch.nn.Dropout(p=0.5),
                                torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
                                torch.nn.ReLU(inplace=True),
                                torch.nn.Linear(in_features=4096, out_features=self.output_classes, bias=True),
                                torch.nn.LogSoftmax(dim=1))


            elif 'resnet' in self.model_arch:
                self.model.fc=torch.nn.Sequential(
                                torch.nn.Linear(in_features=self.in_features, out_features=self.output_classes, bias=True),
                                torch.nn.LogSoftmax(dim=1))

        if self.unfreeze: # This will result in unfreezing and retrain all model layers weights again.
            for param in self.model.parameters():
                param.requires_grad = False




        # Optimizer and Loss Function
        self.loss_function=self.nn_loss_function(type=self.loss_function, **self.loss_function_parameters)
        self.optimizer=self.nn_optimizer(type=self.optimizer, model=self.model, learning_rate=self.learning_rate,  **self.optimizer_parameters)

    def info(self):

        """
        Returns table with all information about the nn_classifier object.

        """

        info=pd.DataFrame.from_dict(({key:str(value) for key, value in self.__dict__.items()}).items())
        info.columns=['Property', 'Value']
        for i in ['train_dataset', 'valid_dataset','test_dataset']:
            if i in self.__dict__.keys():
                info.append({'Property':i+' size', 'Value':len(self.__dict__[i])}, ignore_index=True)
        return info

    def nn_optimizer(self, type, model, learning_rate, **kw):

        """

        Description
        -----------
        Creates an instance of pytorch optimizer


        Parameters
        ----------

        - type (string, required): type of the optimizer. Please see settings for supported optimizers.

        - model (pytorch model, required): model for which optimizer will be used for weight optimization.

        - learning_rate (float, required): learning rate for training.

        - **kw (dictionary, optional): other optional optimizer parameters as per pytorch documentation.

        Returns
        -------
        pytorch nn.optimizer object

        """

        if type not in supported_nn_optimizers:
            log('Error! Optimizer not supported yet. Please check radtorch.settings.supported_nn_optimizers')
            pass
        elif type=='Adam':
            optimizer=torch.optim.Adam(params=model.parameters(),lr=learning_rate, **kw)
        elif type=='AdamW':
            optimizer=torch.optim.AdamW(params=model.parameters(), lr=learning_rate, **kw)
        elif type=='SparseAdam':
            optimizer=torch.optim.SparseAdam(params=model.parameters(), lr=learning_rate, **kw)
        elif type=='Adamax':
            optimizer=torch.optim.Adamax(params=model.parameters(), lr=learning_rate, **kw)
        elif type=='ASGD':
            optimizer=torch.optim.ASGD(params=model.parameters(), lr=learning_rate, **kw)
        elif type=='RMSprop':
            optimizer=torch.optim.RMSprop(params=model.parameters(), lr=learning_rate, **kw)
        elif type=='SGD':
            optimizer=torch.optim.SGD(params=model.parameters(), lr=learning_rate, **kw)
        log('Optimizer selected is '+type)
        return optimizer

    def nn_loss_function(self, type, **kw):
        """

        Description
        -----------
        Creates an instance of pytorch loss function.

        Parameters
        ----------

        - type (string, required): type of the loss function. Please see settings for supported loss functions.

        - **kw (dictionary, optional): other optional loss function parameters as per pytorch documentation.

        Returns
        -------
        pytorch nn.loss_function object

        """

        if type not in supported_nn_loss_functions:
            log('Error! Loss functions not supported yet. Please check radtorch.settings.supported_nn_loss_functions')
            pass
        elif type== 'NLLLoss':
            loss_function=torch.nn.NLLLoss(**kw),
        elif type== 'CrossEntropyLoss':
            loss_function=torch.nn.CrossEntropyLoss(**kw)
        elif type== 'MSELoss':
            loss_function=torch.nn.MSELoss(**kw)
        elif type== 'PoissonNLLLoss':
            loss_function=torch.nn.PoissonNLLLoss(**kw)
        elif type== 'BCELoss':
            loss_function=torch.nn.BCELoss(**kw)
        elif type== 'BCEWithLogitsLoss':
            loss_function=torch.nn.BCEWithLogitsLoss(**kw)
        elif type== 'MultiLabelMarginLoss':
            loss_function=torch.nn.MultiLabelMarginLoss(**kw)
        elif type== 'SoftMarginLoss':
            loss_function=torch.nn.SoftMarginLoss(**kw)
        elif type== 'MultiLabelSoftMarginLoss':
            loss_function=torch.nn.MultiLabelSoftMarginLoss(**kw)
        elif type== 'CosineSimilarity':
            loss_function=torch.nn.CosineSimilarity(**kw)
        log('Loss function selected is '+type)
        return loss_function

    def run(self, **kw):
        """
        Performs Model Training

        Returns
        --------
        Tuple of
            - trained_model: trained neural network model.
            - train_metrics: pandas dataframe of training and validation metrics.
        """

        model=self.model
        train_data_loader=self.train_dataloader
        valid_data_loader=self.valid_dataloader
        train_data_set=self.train_dataset
        valid_data_set=self.valid_dataset
        loss_criterion=self.loss_function
        optimizer=self.optimizer
        epochs=self.epochs
        device=self.device
        if self.lr_scheduler!=None: lr_scheduler=self.lr_scheduler
        else: lr_scheduler=False

        set_random_seed(100)
        start_time=datetime.now()
        training_metrics=[]
        if self.unfreeze:
            log('INFO: unfreeze is set to True. This will unfreeze all model layers and will train from scratch. This might take sometime specially if pre_trained=False.')
        log('Starting training at '+ str(start_time))
        model=model.to(device)
        for epoch in tqdm(range(epochs)):
            epoch_start=time.time()
            # Set to training mode
            model.train()
            # Loss and Accuracy within the epoch
            train_loss=0.0
            train_acc=0.0
            valid_loss=0.0
            valid_acc=0.0
            for i, (inputs, labels, image_paths) in enumerate(train_data_loader):
                # inputs=inputs.float()
                inputs=inputs.to(device)
                labels=labels.to(device)
                # Clean existing gradients
                optimizer.zero_grad()
                # Forward pass - compute outputs on input data using the model
                outputs=model(inputs)
                # Compute loss
                loss=loss_criterion(outputs, labels)
                # Backpropagate the gradients
                loss.backward()
                # Update the parameters
                optimizer.step()
                # Compute the total loss for the batch and add it to train_loss
                train_loss += loss.item() * inputs.size(0)
                # Compute the accuracy
                ret, predictions=torch.max(outputs.data, 1)
                correct_counts=predictions.eq(labels.data.view_as(predictions))
                # Convert correct_counts to float and then compute the mean
                acc=torch.mean(correct_counts.type(torch.FloatTensor))
                # Compute total accuracy in the whole batch and add to train_acc
                train_acc += acc.item() * inputs.size(0)
            # Validation - No gradient tracking needed
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()
                # Validation loop
                for j, (inputs, labels, image_paths) in enumerate(valid_data_loader):
                    inputs=inputs.to(device)
                    labels=labels.to(device)
                    # Forward pass - compute outputs on input data using the model
                    outputs=model(inputs)
                    # Compute loss
                    loss=loss_criterion(outputs, labels)
                    # Compute the total loss for the batch and add it to valid_loss
                    valid_loss += loss.item() * inputs.size(0)
                    # Calculate validation accuracy
                    ret, predictions=torch.max(outputs.data, 1)
                    correct_counts=predictions.eq(labels.data.view_as(predictions))
                    # Convert correct_counts to float and then compute the mean
                    acc=torch.mean(correct_counts.type(torch.FloatTensor))
                    # Compute total accuracy in the whole batch and add to valid_acc
                    valid_acc += acc.item() * inputs.size(0)
            # Find average training loss and training accuracy
            avg_train_loss=train_loss/len(train_data_set)
            avg_train_acc=train_acc/len(train_data_set)
            # Find average validation loss and training accuracy
            avg_valid_loss=valid_loss/len(valid_data_set)
            avg_valid_acc=valid_acc/len(valid_data_set)
            training_metrics.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
            epoch_end=time.time()
            if lr_scheduler:
                lr_scheduler.step(avg_valid_loss)
            log("Epoch : {:03d}/{} : [Training: Loss: {:.4f}, Accuracy: {:.4f}%]  [Validation : Loss : {:.4f}, Accuracy: {:.4f}%] [Time: {:.4f}s]".format(epoch, epochs, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))
        end_time=datetime.now()
        total_training_time=end_time-start_time
        log('Total training time='+ str(total_training_time))
        self.trained_model=model
        self.train_metrics=training_metrics
        self.train_metrics = pd.DataFrame(data=self.train_metrics, columns = ['Train_Loss', 'Valid_Loss', 'Train_Accuracy', 'Valid_Accuracy'])
        return self.trained_model, self.train_metrics

    def confusion_matrix(self, target_dataset=None, figure_size=(8,6), cmap=None):

        """
        Displays confusion matrix for trained nn_classifier on test dataset.

        Parameters
        ----------

        - target_dataset (pytorch dataset, optional): this option can be used to test the trained model on an external test dataset. If set to None, the confusion matrix is generated using the test dataset initially specified in the data_processor. default=None.

        - figure_size (tuple, optional): size of the figure as width, height. default=(8,6)

        """

        if target_dataset==None:target_dataset=self.test_dataset
        target_classes=(self.data_processor.classes()).keys()

        show_nn_confusion_matrix(model=self.trained_model, target_data_set=target_dataset, target_classes=target_classes, device=self.device, figure_size=figure_size, cmap=cmap)

    def roc(self, **kw):
        """
        Displays ROC and AUC of trained model with test dataset

        """
        show_roc([self], **kw)

    def metrics(self, figure_size=(700,400)):

        """
        Displays graphical representation of train/validation loss /accuracy.

        Parameters
        ----------

        - figure_size (tuple, optional): size of the figure as width, height. default=(700,400)

        """

        show_metrics([self], figure_size=figure_size)

    def predict(self,  input_image_path, all_predictions=True, **kw):

        """
        Description
        -----------
        Displays classs prediction for a target image using a trained classifier.


        Parameters
        ----------

        - input_image_path (string, required): path to target image.

        - all_predictions (boolean, optional): True to display prediction percentage accuracies for all prediction classes. default=True.

        """

        model=self.trained_model
        transformations=self.transformations

        if input_image_path.endswith('dcm'):
            target_img=dicom_to_pil(input_image_path)
        else:
            target_img=Image.open(input_image_path).convert('RGB')

        target_img_tensor=transformations(target_img)
        target_img_tensor=target_img_tensor.unsqueeze(0)

        with torch.no_grad():
            model.to('cpu')
            target_img_tensor.to('cpu')
            model.eval()
            out=model(target_img_tensor)
            softmax=torch.exp(out).cpu()
            prediction_percentages=softmax.cpu().numpy()[0]
            # prediction_percentages=[i*100 for i in prediction_percentages]
            prediction_percentages = [("%.4f" % x) for x in prediction_percentages]
            _, final_prediction=torch.max(out, 1)
            prediction_table=pd.DataFrame(list(zip(self.data_processor.classes().keys(), [*range(0, len(prediction_percentages), 1)], prediction_percentages)), columns=['label','label_idx', 'prediction_accuracy'])

        if all_predictions:
            return prediction_table
        else:
            return final_prediction.item(), prediction_percentages[final_prediction.item()]

    def misclassified(self, num_of_images=4, figure_size=(5,5), table=False, **kw):

        """
        Description
        -----------
        Displays sample of images misclassified by the classifier from test dataset.


        Parameters
        ----------

        - num_of_images (integer, optional): number of images to be displayed. default=4.

        - figure_size (tuple, optional): size of the figure as width, height. default=(5,5).

        - table (boolean, optional): True to display a table of all misclassified images including image path, true label and predicted label.

        """
        misclassified_table = show_nn_misclassified(model=self.trained_model, target_data_set=self.test_dataset, num_of_images=num_of_images, device=self.device, transforms=self.data_processor.transformations, is_dicom = self.is_dicom, figure_size=figure_size)
        if table:
            return misclassified_table



# NEEDS TESTING
class Feature_Selector(Classifier):

    def feature_feature_correlation(self, cmap='Blues', figure_size=(20,15)):
        corrmat = self.features.corr()
        f, ax = plt.subplots(figsize=figure_size)
        sns.heatmap(corrmat, cmap=cmap, linewidths=.1,ax=ax)

    def feature_label_correlation(self, threshold=0.5):
        corrmat = self.feature_table.corr()
        corr_target = abs(corrmat[self.label_column])
        relevant_features = corr_target[corr_target>threshold]
        df = pd.DataFrame(relevant_features)
        df.columns=['Score']
        df.index.rename('Feature')
        best_features_scores=df.sort_values(by=['Score'], ascending=False)
        best_features_names=df.index.tolist()
        best_features_names.remove(self.label_column)
        best_features_table=self.feature_table[df.index.tolist()]
        return best_features_scores, best_features_names, best_features_table

    def univariate(self, test='chi2', num_features=20):
        if test=='chi2':
          selector = SelectKBest(chi2, k=num_features)
        elif test=='anova':
          selector = SelectKBest(f_classif, k=num_features)
        elif test=='mutual_info':
          selector = SelectKBest(mutual_info_classif, k=num_features)
        selector.fit(self.train_features, self.train_labels)
        feature_score=selector.scores_.tolist()
        df=pd.DataFrame(list(zip(self.feature_names, feature_score)), columns=['Feature', 'Score'])
        best_features_scores=df.sort_values(by=['Score'], ascending=False)[:num_features]
        best_features_names=best_features_scores.Feature.tolist()
        best_features_table=self.feature_table[best_features_names+[self.label_column]]
        return best_features_scores, best_features_names, best_features_table

    def variance(self, threshold=0, num_features=20):
        selector=VarianceThreshold(threshold=threshold)
        selector.fit(self.train_features, self.train_labels)
        feature_score=selector.variances_.tolist()
        df=pd.DataFrame(list(zip(self.feature_names, feature_score)), columns=['Feature', 'Score'])
        best_features_scores=df.sort_values(by=['Score'], ascending=False)[:num_features]
        best_features_names=best_features_scores.Feature.tolist()
        best_features_table=self.feature_table[best_features_names+[self.label_column]]
        return best_features_scores, best_features_names, best_features_table

    def rfe(self, step=1, rfe_features=None):
        if 'rfe_feature_rank' not in self.__dict__.keys():
          self.selector=RFE(estimator=self.classifier, n_features_to_select=rfe_features, step=step)
          self.selector.fit(self.train_features, self.train_labels)
          self.rfe_feature_rank=self.selector.ranking_
        df= pd.DataFrame(list(zip(self.feature_names, self.rfe_feature_rank.tolist())), columns=['Feature', 'Rank'])
        best_features_names=[x for x,v in list(zip(G.feature_names, G.selector.support_.tolist())) if v==True]
        best_features_scores=df.sort_values(by=['Rank'], ascending=True)
        best_features_table=self.feature_table[best_features_names+[self.label_column]]
        return best_features_scores, best_features_names, best_features_table

    def rfecv(self, step=1, n_jobs=-1, verbose=0):
        self.rfecv_selector=RFECV(estimator=self.classifier, step=step, cv=StratifiedKFold(self.num_splits),scoring='accuracy', n_jobs=-1, verbose=verbose)
        self.rfecv_selector.fit(self.train_features, self.train_labels)
        self.optimal_feature_number=self.rfecv_selector.n_features_
        self.optimal_features_names=[x for x,v in list(zip(self.feature_names, self.rfecv_selector.support_.tolist())) if v==True]
        self.best_features_table=self.feature_table[self.optimal_features_names+[self.label_column]]
        log('Optimal Number of Features = '+ str(self.optimal_feature_number))
        j = range(1, len(self.rfecv_selector.grid_scores_) + 1)
        i = self.rfecv_selector.grid_scores_
        output_notebook()
        p = figure(plot_width=600, plot_height=400)
        p.line(j, i, line_width=2, color='#1A5276')
        p.line([self.optimal_feature_number]*len(i),i,line_width=2, color='#F39C12', line_dash='dashed')
        p.xaxis.axis_line_color = '#D6DBDF'
        p.yaxis.axis_line_color = '#D6DBDF'
        p.xgrid.grid_line_color=None
        p.yaxis.axis_line_width = 2
        p.xaxis.axis_line_width = 2
        p.xaxis.axis_label = 'Number of features selected. Optimal = '+str(self.optimal_feature_number)
        p.yaxis.axis_label = 'Cross validation score (nb of correct classifications)'
        p.xaxis.major_tick_line_color = '#D6DBDF'
        p.yaxis.major_tick_line_color = '#D6DBDF'
        p.xaxis.minor_tick_line_color = '#D6DBDF'
        p.yaxis.minor_tick_line_color = '#D6DBDF'
        p.yaxis.major_tick_line_width = 2
        p.xaxis.major_tick_line_width = 2
        p.yaxis.minor_tick_line_width = 0
        p.xaxis.minor_tick_line_width = 0
        p.xaxis.major_label_text_color = '#99A3A4'
        p.yaxis.major_label_text_color = '#99A3A4'
        p.outline_line_color = None
        p.toolbar.autohide = True
        p.title.text='Recursive Feature Elimination with '+str(self.num_splits)+'-split Cross Validation'
        p.title_location='above'
        show(p)
        return self.optimal_features_names, self.best_features_table

    def tsne(self, feature_table=None, figure_size=(800, 800), colormap=COLORS3, **kwargs):
        if isinstance(feature_table, pd.DataFrame):
            y = feature_table
        else:
            y = self.feature_table[self.feature_names+[self.label_column]]
        tsne = TSNE(n_components=2, random_state=0)
        X_2d = tsne.fit_transform(y)
        output_notebook()
        p = figure(tools=TOOLS, plot_width=figure_size[0], plot_height=figure_size[1])
        for i in y.label_idx.unique().tolist():
            p.scatter(X_2d[y[self.label_column] == i, 0], X_2d[y[self.label_column] == i, 1], radius=0.4, fill_alpha=0.6,line_color=None, fill_color=colormap[i])
        p.xaxis.axis_line_color = '#D6DBDF'
        p.yaxis.axis_line_color = '#D6DBDF'
        p.xgrid.grid_line_color=None
        p.ygrid.grid_line_color=None
        p.yaxis.axis_line_width = 2
        p.xaxis.axis_line_width = 2
        p.xaxis.major_tick_line_color = '#D6DBDF'
        p.yaxis.major_tick_line_color = '#D6DBDF'
        p.xaxis.minor_tick_line_color = '#D6DBDF'
        p.yaxis.minor_tick_line_color = '#D6DBDF'
        p.yaxis.major_tick_line_width = 2
        p.xaxis.major_tick_line_width = 2
        p.yaxis.minor_tick_line_width = 0
        p.xaxis.minor_tick_line_width = 0
        p.xaxis.major_label_text_color = '#99A3A4'
        p.yaxis.major_label_text_color = '#99A3A4'
        p.outline_line_color = None
        p.toolbar.autohide = True
        p.title.text='t-distributed Stochastic Neighbor Embedding (t-SNE)'
        p.title_location='above'
        show(p)
