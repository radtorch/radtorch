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

from ..settings import *
from .dicom import  *
from .utils import *
from .general import *

def load_predefined_datatables(*args, **kwargs):

    """
    Description
    -----------
    Creates a dictionary of datasets from a dictionary of predfined tables for each of train/valid/test


    Parameters
    -----------
    predefined_datasets (dictionary, required): Dictionary of tables containing data for train/valid/test as {'train':, 'valid':, 'test':}
    Rest of parameters as with "Dataset_from_table"

    Returns
    -----------
    Dictionary of dataset objects as {'train':train_dataset, 'valid':valid_dataset, 'test':test_dataset}
    """

    train_dataset=Dataset_from_table(**kwargs, table=kwargs['predefined_datasets']['train'])
    valid_dataset=Dataset_from_table(**kwargs, table=kwargs['predefined_datasets']['valid'])
    test_dataset=Dataset_from_table(**kwargs, table=kwargs['predefined_datasets']['test'])
    output={'train':train_dataset, 'valid':valid_dataset, 'test':test_dataset}
    return output


def over_sample(dataset, shuffle=True, **kwargs):
    balanced_dataset = copy.deepcopy(dataset)
    max_size = balanced_dataset.input_data[balanced_dataset.image_label_column].value_counts().max()
    lst = [balanced_dataset.input_data]
    for class_index, group in balanced_dataset.input_data.groupby(balanced_dataset.image_label_column):
      lst.append(group.sample(max_size-len(group), replace=True))
    balanced_dataframe = pd.concat(lst)
    if shuffle:
        balanced_dataframe = balanced_dataframe.sample(frac=1).reset_index(drop=True)
    balanced_dataset.input_data = balanced_dataframe
    return balanced_dataset


def calculate_mean_std(dataloader):
    '''
    Source
    -------
    https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/6
    '''
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, labels, paths in tqdm(dataloader, total=len(dataloader)):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
    mean /= nb_samples
    std /= nb_samples
    return (mean, std)


# def balance_dataset(dataset, label_col, upsample=True):
#   balanced_dataset=copy.deepcopy(dataset)
#   df = balanced_dataset.input_data
#   counts=df.groupby(label_col).count()
#   classes=df[label_col].unique().tolist()
#   max_class_num=counts.max()[0]
#   max_class_id=counts.idxmax()[0]
#   min_class_num=counts.min()[0]
#   min_class_id=counts.idxmin()[0]
#   if upsample:
#     resampled_subsets = [df[df[label_col]==max_class_id]]
#     for i in [x for x in classes if x != max_class_id]:
#       class_subset=df[df[label_col]==i]
#       upsampled_subset=resample(class_subset, n_samples=max_class_num, random_state=100)
#       resampled_subsets.append(upsampled_subset)
#   else:
#     resampled_subsets = [df[df[label_col]==min_class_id]]
#     for i in [x for x in classes if x != min_class_id]:
#       class_subset=df[df[label_col]==i]
#       upsampled_subset=resample(class_subset, n_samples=min_class_num, random_state=100)
#       resampled_subsets.append(upsampled_subset)
#   resampled_df = pd.concat(resampled_subsets)
#   balanced_dataset.input_data=resampled_df
#   return balanced_dataset


def split_dataset(dataset, valid_percent=0.2, test_percent=0.2, equal_class_split=True, shuffle=True, sample=False,  **kwargs):
    num_all = len(dataset)
    train_percent = 1.0 - (valid_percent+test_percent)
    num_classes = dataset.input_data[dataset.image_label_column].unique()
    classes_df = []
    for i in num_classes:
        temp_df = dataset.input_data.loc[dataset.input_data[dataset.image_label_column]==i]
        if shuffle:
          temp_df = temp_df.sample(frac=1).reset_index(drop=True)
        train, validate, test = np.split(temp_df.sample(frac=1), [int(train_percent*len(temp_df)), int((train_percent+valid_percent)*len(temp_df))])
        if isinstance(sample, float):
            train = train.sample(frac=sample)
            validate = validate.sample(frac=sample)
            test = test.sample(frac=sample)
        classes_df.append((train, validate, test))
    output = {}
    train_df = (pd.concat([i[0] for i in classes_df])).sample(frac=1).reset_index(drop=True)
    valid_df = (pd.concat([i[1] for i in classes_df])).sample(frac=1).reset_index(drop=True)
    output['train'] =  Dataset_from_table(data_directory=dataset.data_directory,is_dicom=dataset.is_dicom, table=train_df, mode=dataset.mode, wl=dataset.wl, transformations=dataset.transformations)
    output['valid'] =  Dataset_from_table(data_directory=dataset.data_directory,is_dicom=dataset.is_dicom, table=valid_df, mode=dataset.mode, wl=dataset.wl, transformations=dataset.transformations)
    if test_percent != 0:
        test_df = (pd.concat([i[2] for i in classes_df])).sample(frac=1).reset_index(drop=True)
        output['test'] =Dataset_from_table(data_directory=dataset.data_directory,is_dicom=dataset.is_dicom, table=test_df, mode=dataset.mode, wl=dataset.wl, transformations=dataset.transformations)
    return  output


def set_random_seed(seed):
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        log('Random seed '+str(seed)+' set successfully')
    except:
        raise TypeError('Error. Could not set Random Seed. Please check again.')
        pass


def list_of_files(root):
    listOfFile = os.listdir(root)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(root, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + list_of_files(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles


def path_to_class(filepath):
    item_class = (Path(filepath)).parts
    return item_class[-2]


def root_to_class(root):
    classes = [d.name for d in os.scandir(root) if d.is_dir()]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def class_to_idx(classes):
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return class_to_idx


def datatable_from_filepath(*filelist,classes:list): #KareemElFatairy
    """ purpose: Create dataframe of file pathes and labels extracted from supplied folders.
        Argument:
        *filelist: returns list of paths.
        classes: a list of desired classes as seen in file name.
    """
    file_lists = map(list_of_files,filelist)  #get a list of files from folders
    data={'IMAGE_PATH':[],'IMAGE_LABEL':[]}
    for file_list in file_lists:
      for file_path in file_list: #create lists of files with the specified label and append to the dictionary
        for item in classes:
          if item.casefold() in file_path.casefold():   #case insensitive match
            data['IMAGE_PATH'].append(file_path)
            data['IMAGE_LABEL'].append(item)
    df=pd.DataFrame(data)
    return df


def parse_voc_xml(label_file):
    '''

    '''
    output = {}

    label_data=xmltodict.parse(open(label_file , 'rb'))

    output['width']=int(label_data['annotation']['size']['width'])
    output['height']=int(label_data['annotation']['size']['height'])
    output['depth']=int(label_data['annotation']['size']['depth'])
    output['x_min']=int(label_data['annotation']['object']['bndbox']['xmin'])
    output['x_max']=int(label_data['annotation']['object']['bndbox']['xmax'])
    output['y_min']=int(label_data['annotation']['object']['bndbox']['ymin'])
    output['y_max']=int(label_data['annotation']['object']['bndbox']['ymax'])
    output['image_id']=label_data['annotation']['filename']
    output['labels']=label_data['annotation']['object']['name']
    output['area']= (output['x_max']-output['x_min'])*(output['y_max']-output['y_min'])

    return output
