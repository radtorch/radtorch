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

from ..settings import *
from ..utils import *

from .dataset import *



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
                data_type='image_classification',
                format='voc',
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
        self.data_type=data_type
        self.format=format
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
        else:
            if self.data_type=='object_detection':
                if self.format=='voc':
                    box_files=[x for x in list_of_files(self.data_directory) if x.endswith('.xml')]
                    parsed_data=[]
                    for i in box_files:
                        parsed_data.append(parse_voc_xml(i))
                    self.table=pd.DataFrame(parsed_data)
                    self.table[self.image_path_column]=self.table['image_id']
                    self.table[self.image_label_column]=self.table['labels']
                    self.is_path=False
            else:
                self.table=create_data_table(directory=self.data_directory, is_dicom=self.is_dicom, image_path_column=self.image_path_column, image_label_column=self.image_label_column)


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
            if len(mean) != 3 or len(std) != 3:
                log ('Error! Shape of supplied mean and/or std does not equal 3. Please check that the mean/std follow the following format: ((mean, mean, mean), (std, std, std))')
                pass
            else:
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
                                            transformations=self.transformations,
                                            data_type=self.data_type,
                                            format=self.format
                                            )

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
                                                transformations=self.train_transformations,
                                                data_type=self.data_type,
                                                format=self.format
                                                )
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
                                                transformations=self.transformations,
                                                data_type=self.data_type,
                                                format=self.format
                                                )
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
                                                transformations=self.train_transformations,
                                                data_type=self.data_type,
                                                format=self.format
                                                )
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
                                            transformations=self.transformations,
                                            data_type=self.data_type,
                                            format=self.format
                                            )

        self.master_dataloader=torch.utils.data.DataLoader(dataset=self.master_dataset,batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.train_dataloader=torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_dataloader=torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def classes(self):
        """
        Returns dictionary of classes/class_idx in data.
        """
        return self.master_dataset.class_to_idx

    def class_table(self):
        """
        Returns table of classes/class_idx in data.
        """
        return pd.DataFrame(list(zip(self.master_dataset.class_to_idx.keys(), self.master_dataset.class_to_idx.values())), columns=['Label', 'Label_idx'])

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
                print (k)
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
