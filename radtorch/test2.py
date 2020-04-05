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
from radtorch.model import *
from radtorch.data import *
from radtorch.vis import *
from radtorch.general import *
from radtorch.dataset import *



class Pipeline():

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        for k, v  in self.DEFAULT_SETTINGS.items():
            if k not in kwargs.keys():
                setattr(self, k, v)

        if 'device' not in kwargs.keys():
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data_subsets = ['dataset', 'train_dataset', 'valid_dataset', 'test_dataset']

    def info(self):
        info = pd.DataFrame.from_dict(({key:str(value) for key, value in self.__dict__.items()}).items())
        info.columns = ['Property', 'Value']
        for i in self.data_subsets :
            if i in self.__dict__.keys(): info = info.append({'Property':i, 'Value':len(self.__dict__[i])}, ignore_index=True)
        return info

    def dataset_info(self, plot=True, fig_size=(500,300)):
        info_dict = {}
        for i in self.data_subsets :
            if i in self.__dict__.keys():
                info_dict[i] = show_dataset_info(self.__dict__[i])
                info_dict[i].style.set_caption(i)
        if plot:
            plot_dataset_info(info_dict, plot_size= fig_size)
        else:
            for k, v in info_dict.items():
                display(v)

    def sample(self, figure_size=(10,10), show_labels=True, show_file_name=False):
        show_dataloader_sample(self.dataloader, figure_size=figure_size, show_labels=show_labels, show_file_name = show_file_name)

    def metrics(self, figure_size=(500,300)):
        return show_metrics(self.classifiers,  fig_size=figure_size)

    def export(self, output_path):
        try:
            outfile = open(output_path,'wb')
            pickle.dump(self,outfile)
            outfile.close()
            print ('Pipeline exported successfully.')
        except:
            raise TypeError('Error! Pipeline could not be exported.')

    def classes(self):
        return self.dataset.class_to_idx


class Image_Classification(Pipeline):

    def __init__(self, **kwargs):
        super(Image_Classification, self).__init__(**kwargs, DEFAULT_SETTINGS=IMAGE_CLASSIFICATION_PIPELINE_SETTINGS)
        self.classifiers = [self]

        # Create Initial Master Dataset
        if isinstance(self.table, pd.DataFrame): self.dataset=Dataset_from_table(**kwargs)
        else: self.dataset=Dataset_from_folder(**kwargs)
        self.num_output_classes = len(self.dataset.classes)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        # Custom Resize Adjustement
        if isinstance(self.custom_resize, bool): self.resize = model_dict[self.model_arch]['input_size']
        elif isinstance(self.custom_resize, int): self.resize = self.custom_resize

        # Create transformations
        if self.is_dicom:
            self.transformations = transforms.Compose([
                    transforms.Resize((self.resize, self.resize)),
                    transforms.transforms.Grayscale(3),
                    transforms.ToTensor()])
        else:
            self.transformations = transforms.Compose([
                transforms.Resize((self.resize, self.resize)),
                transforms.ToTensor()])

        # Calculate Normalization if required
        if self.normalize=='auto':
            mean, std = self.dataset.mean_std()
            self.transformations.transforms.append(transforms.Normalize(mean=mean, std=std))
        elif isinstance (self.normalize, tuple):
            mean, std = self.normalize
            self.transformations.transforms.append(transforms.Normalize(mean=mean, std=std))

        # Recreate Transformed Master Dataset
        if isinstance(self.table, pd.DataFrame): self.dataset=Dataset_from_table(**kwargs, transformations=self.transformations)
        else: self.dataset=Dataset_from_folder(**kwargs, transformations=self.transformations)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        # Split Master Dataset
        self.dataset_dictionary = self.dataset.split(valid_percent=self.valid_percent, test_percent=self.test_percent)

        # Create train/valid/test datasets and dataloaders
        for k, v in self.dataset_dictionary.items():
            if self.balance_class: setattr(self, k+'_dataset', v.balance())
            else: setattr(self, k+'_dataset', v)
            setattr(self, k+'_dataloader', torch.utils.data.DataLoader(dataset=self.__dict__[k+'_dataset'], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers))

        self.train_model = create_model(output_classes=self.num_output_classes,mode = 'train', model_arch=self.model_arch,pre_trained=self.pre_trained, unfreeze_weights=self.unfreeze_weights )

        self.train_model = self.train_model.to(self.device)

        if self.loss_function in supported_image_classification_losses:
            self.loss_function = create_loss_function(self.loss_function)
        else:
            raise TypeError('Selected loss function is not supported with image classification pipeline. Please use modelsutils.supported() to view list of supported loss functions.')
            pass

        if self.optimizer in supported_optimizer:
            self.optimizer = create_optimizer(traning_model=self.train_model, optimizer_type=self.optimizer)
        else:
            raise TypeError('Selected optimizer is not supported with image classification pipeline. Please use modelsutils.supported() to view list of supported optimizers.')
            pass


    def run(self, verbose=True):
        try:
            print ('Starting Image Classification Pipeline Training')
            self.trained_model, self.train_metrics = train_model(
                                                    model = self.train_model,
                                                    train_data_loader = self.train_dataloader,
                                                    valid_data_loader = self.valid_dataloader,
                                                    train_data_set = self.train_dataset,
                                                    valid_data_set = self.valid_dataset,
                                                    loss_criterion = self.loss_function,
                                                    optimizer = self.optimizer,
                                                    epochs = self.train_epochs,
                                                    device = self.device,
                                                    verbose=verbose)
            self.train_metrics = pd.DataFrame(data=self.train_metrics, columns = ['Train_Loss', 'Valid_Loss', 'Train_Accuracy', 'Valid_Accuracy'])
        except:
            raise TypeError('Could not train image classification pipeline. Please check provided parameters.')
            pass













##
