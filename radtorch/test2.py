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

        for K, V in self.DEFAULT_SETTINGS.items():
            if K not in kwargs.keys():
                setattr(self, K, V)



    def info(self):
        info = pd.DataFrame.from_dict(({key:str(value) for key, value in self.__dict__.items()}).items())
        info.columns = ['Property', 'Value']
        for i in ['train_dataset', 'valid_dataset', 'test_dataset']:
            if i in in self.__dict__.keys(): info = info.append({'Property':i, 'Value':len(self.__dict__[i])}, ignore_index=True)
        return info


    def dataset_info(self, plot=True, fig_size=(500,300)):
        info_dict = {}
        for i in ['dataset', 'train_dataset', 'valid_dataset', 'test_dataset']:
            if i in in self.__dict__.keys():
                info_dict[i] = show_dataset_info(self.__dict__[i])
                info_dict[i].style.set_caption(i)
        if plot:
            plot_dataset_info(info_dict, plot_size= fig_size)
        else:
            for k, v in info_dict.items():
                display(v)


    def sample(self, fig_size=(10,10), show_labels=True, show_file_name=False):
        if 'train_dataloader' in self.__dict__.keys():
            displayed_dataloader=self.train_dataloader
            displayed_dataset=self.train_dataset
        else:
            displayed_dataloader=self.dataloader
            displayed_dataset=self.dataset
            batch = next(iter(displayed_dataloader))
            images, labels, paths = batch
            images = [np.moveaxis(x, 0, -1) for x in images.numpy()]
            if show_labels:
              titles = labels.numpy()
              titles = [((list(displayed_dataset.class_to_idx.keys())[list(displayed_dataset.class_to_idx.values()).index(i)]), i) for i in titles]
            if show_file_name:
              titles = [ntpath.basename(x) for x in paths]

            show_dataloader_sample

            plot_images(images=images, titles=titles, figure_size=fig_size)
