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
from radtorch.settings import *
from radtorch.dicomutils import  *
from radtorch.visutils import *
from radtorch.settings import *
from radtorch.datautils import *




class RADTorch_Dataset(Dataset):
    def __init__(self, **kwargs): #defines the default parameters for dataset class.
        for k,v in kwargs.items():
            setattr(self, k, v)

        for k, v in DEFAULT_DATASET_SETTINGS.items():
            if k not in kwargs.keys():
                setattr(self, k, v)

    def __getitem__(self, index): #handles how to get an image of the dataset.
        image_path = self.input_data.iloc[index][self.image_path_column]
        if self.is_dicom:
            image = dicom_to_narray(image_path, self.mode, self.wl)
            image = Image.fromarray(image)

        else:
            image = Image.open(image_path).convert('RGB')

        image = self.transformations(image)

        if self.multi_label == True:
            label = self.input_data.iloc[index][self.image_label_column]
            label_idx = self.input_data.iloc[index]['MULTI_LABEL_IDX']

        else:
            label = self.input_data.iloc[index][self.image_label_column]
            label_idx = [v for k, v in self.class_to_idx.items() if k == label][0]

        return image, label_idx, image_path

    def __len__(self): #returns number of images in dataset.
        return len(self.dataset_files)

    def info(self): #returns table of dataset information.
        return show_dataset_info(self)

    def classes(self): #returns list of classes in dataset.
        return self.classes

    def class_to_idx(self): #returns mapping of classes to class id (dictionary).
        return self.class_to_idx

    def parameters(self): #returns all the parameter names of the dataset.
        return self.__dict__.keys()

    def split(self, **kwargs): #splits dataset into train/valid/split, takes test_percent and valid_percent.
        return split_dataset(dataset=self, **kwargs)

    def balance(self, **kwargs): #solves class imbalance in dataset through over-sampling of classes with less images.
        return over_sample(dataset=self, **kwargs)

    def mean_std(self): #calculates mean and standard deviation of dataset.
        self.mean, self.std =  calculate_mean_std(torch.utils.data.DataLoader(dataset=self))
        return tuple(self.mean.tolist()), tuple(self.std.tolist())

    def normalized(self, **kwargs): #retruns a normalized dataset with either mean/std of the dataset or a user specified mean/std
        if 'mean' in kwargs.keys() and 'std' in kwargs.keys():
            mean = kwargs['mean']
            std = kwargs['std']
        else:
            mean, std = self.mean_std()
        normalized_dataset = copy.deepcopy(self)
        normalized_dataset.transformations.transforms.append(transforms.Normalize(mean=mean, std=std))
        return normalized_dataset


class Dataset_from_table(RADTorch_Dataset):
    def __init__(self, **kwargs):
        super(Dataset_from_table, self).__init__(**kwargs)
        if isinstance(self.table, pd.DataFrame):
            self.input_data = self.table
        elif self.table != None:
            self.input_data = pd.read_csv(self.input_source)
        else:
            raise TypeError('Error! No label table was selected. Please check.')
        if self.is_dicom: self.dataset_files = [x for x in (self.input_data[self.image_path_column].tolist()) if x.endswith('.dcm')]
        else: self.dataset_files = [x for x in (self.input_data[self.image_path_column].tolist()) if x.endswith(IMG_EXTENSIONS)]
        if self.multi_label == True:
            self.classes = list(np.unique([item for t in self.input_data[self.image_label_column].to_numpy() for item in t]))
            self.class_to_idx = class_to_idx(self.classes)
            self.multi_label_idx = []
            for i, row in self.input_data.iterrows():
                t = []
                for u in self.classes:
                    if u in row[self.image_label_column]:
                        t.append(1)
                    else:
                        t.append(0)
                self.multi_label_idx.append(t)
            self.input_data['MULTI_LABEL_IDX'] = self.multi_label_idx
        else:
            self.classes =  list(self.input_data[self.image_label_column].unique())
            self.class_to_idx = class_to_idx(self.classes)
        if len(self.dataset_files)==0:
            print ('Error! No data files found in directory:', self.data_directory)

        if len(self.classes)    ==0:
            print ('Error! No classes extracted from directory:', self.data_directory)


class Dataset_from_folder(RADTorch_Dataset):
    def __init__(self, **kwargs):
        super(Dataset_from_folder, self).__init__(**kwargs)
        self.classes, self.class_to_idx = root_to_class(self.data_directory)
        self.all_files = list_of_files(self.data_directory)
        if self.is_dicom: self.dataset_files = [x for x in self.all_files  if x.endswith('.dcm')]
        else: self.dataset_files = [x for x in self.all_files if x.endswith(IMG_EXTENSIONS)]
        self.all_classes = [path_to_class(i) for i in self.dataset_files]
        self.input_data = pd.DataFrame(list(zip(self.dataset_files, self.all_classes)), columns=[self.image_path_column, self.image_label_column])
        if len(self.dataset_files)==0:
            print ('Error! No data files found in directory:', self.data_directory)
        if len(self.classes)==0:
            print ('Error! No classes extracted from directory:', self.data_directory)


def load_predefined_datatables(*args, **kwargs):
    train_dataset = Dataset_from_table(kwargs, table=kwargs['predefined_datasets']['train'])
    valid_dataset = Dataset_from_table(kwargs, table=kwargs['predefined_datasets']['valid'])
    test_dataset = Dataset_from_table(kwargs, table=kwargs['predefined_datasets']['test'])

    return train_dataset, valid_dataset, test_dataset
