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

        data_subsets = ['dataset', 'train_dataset', 'valid_dataset', 'test_dataset']

    def info(self):
        info = pd.DataFrame.from_dict(({key:str(value) for key, value in self.__dict__.items()}).items())
        info.columns = ['Property', 'Value']
        for i in data_subsets:
            if i in in self.__dict__.keys(): info = info.append({'Property':i, 'Value':len(self.__dict__[i])}, ignore_index=True)
        return info

    def dataset_info(self, plot=True, fig_size=(500,300)):
        info_dict = {}
        for i in data_subsets:
            if i in in self.__dict__.keys():
                info_dict[i] = show_dataset_info(self.__dict__[i])
                info_dict[i].style.set_caption(i)
        if plot:
            plot_dataset_info(info_dict, plot_size= fig_size)
        else:
            for k, v in info_dict.items():
                display(v)

    def sample(self, figure_size=(10,10), show_labels=True, show_file_name=False):
        if 'train_dataloader' in self.__dict__.keys():
            displayed_dataloader=self.train_dataloader
        else:
            displayed_dataloader=self.dataloader
            show_dataloader_sample(displayed_dataloader, figure_size=figure_size, show_labels=show_labels, show_file_name = show_file_name)

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



class Image_Classification(Pipeline):
    def __init__(self, DEFAULT_SETTINGS=IMAGE_CLASSIFICATION_PIPELINE_SETTINGS, **kwargs):
        super(Image_Classification, self).__init__(**kwargs)

        if self.table: self.dataset=Dataset_from_table(kwargs)
        else: self.dataset=Dataset_from_folder(kwargs)
        self.num_output_classes = len(self.dataset.classes)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset,kwargs)
