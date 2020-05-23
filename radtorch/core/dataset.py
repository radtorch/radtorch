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
                data_type='image_classification',
                format='voc',
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
        self.data_type=data_type
        self.format=format

        # Create Data Table
        if isinstance(self.table, pd.DataFrame): self.input_data=self.table
        elif isinstance(self.table, string): self.input_data=pd.read_csv(self.table)
        else:
            if self.data_type=='object_detection':
                if self.format=='voc':
                    box_files=[x for x in list_of_files(self.data_directory) if x.endswith('.xml')]
                    parsed_data=[]
                    for i in box_files:
                        parsed_data.append(parse_voc_xml(i))
                    self.input_data=pd.DataFrame(parsed_data)
                    self.input_data[self.image_path_column]=self.input_data['image_id']
                    self.input_data[self.image_label_column]=self.input_data['labels']
                    self.is_path=False
            else:
                self.input_data=create_data_table(data_directory=self.data_directory, is_dicom=self.is_dicom, image_path_column=self.image_path_column, image_label_column=self.image_label_column)

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
        if self.data_type=='object_detection':
            self.class_to_idx={k:v+1 for k, v in self.class_to_idx.items()} # Add 1 so that the first class will not be zero
            if 'background' in self.class_to_idx.keys():
                self.class_to_idx['background']=0


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
        if self.data_type=='image_classification':
            return image, label_idx, image_path
        elif self.data_type=='object_detection':
            target={}
            boxes=[self.input_data.iloc[index]['x_min'], self.input_data.iloc[index]['x_max'], self.input_data.iloc[index]['y_min'], self.input_data.iloc[index]['y_max']]
            target['boxes']=torch.as_tensor(boxes, dtype=torch.float32)
            label=[v for k, v in self.class_to_idx.items() if k == label][0]
            target['labels']=torch.tensor([label], dtype=torch.int64)
            target['area']=self.input_data.iloc[index]['area']
            target['image_id']=torch.tensor([self.input_data.iloc['image_id']])
            target['iscrowd']=torch.zeros[1]
            return image, target


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
