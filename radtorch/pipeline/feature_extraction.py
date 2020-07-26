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

from ..settings import *
from ..core import *
from ..utils import *


class Feature_Extraction():

    def __init__(self,
                data_directory,
                table=None,
                is_dicom=True,
                normalize=((0, 0, 0), (1, 1, 1)),
                balance_class=False,
                batch_size=16,
                num_workers=0,
                model_arch='resnet50',
                custom_resize=False,
                pre_trained=True,
                unfreeze=False,
                label_column='IMAGE_LABEL',
                **kwargs):

        self.data_directory=data_directory
        self.table=table
        self.is_dicom=is_dicom
        self.normalize=normalize
        self.balance_class=balance_class
        self.batch_size=batch_size
        self.num_workers=num
        self.model_arch=model_arch
        self.custom_resize=custom_resize
        self.pre_trained=pre_trained
        self.unfreeze=unfreeze
        self.label_column=label_column

        for k, v in kwargs.items():
            setattr(self, k, v)

        if 'device' not in kwargs.keys(): self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_processor=Data_Processor(**self.__dict__)
        self.feature_extractor=Feature_Extractor(dataloader=self.data_processor.dataloader, **self.__dict__)

    def info(self):
        info=pd.DataFrame.from_dict(({key:str(value) for key, value in self.__dict__.items()}).items())
        info.columns=['Property', 'Value']
        return info

    def run(self, **kw):
        set_random_seed(100)
        if 'feature_table' in kw.keys():
            log('Loading Extracted Features')
            self.feature_table=kw['feature_table']
            self.feature_names=kw['feature_names']
        elif 'feature_table' not in self.__dict__.keys():
            log('Running Feature Extraction.')
            self.feature_extractor.run()
            self.feature_table=self.feature_extractor.feature_table
            self.feature_names=self.feature_extractor.feature_names
        return self.feature_table

    def export(self, output_path):
        try:
            outfile=open(output_path,'wb')
            pickle.dump(self,outfile)
            outfile.close()
            log('Pipeline exported successfully.')
        except:
            raise TypeError('Error! Pipeline could not be exported.')
