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

## Code Last Updated: 08/01/2020

from ..settings import *
from ..core import *
from ..utils import *
# from ..beta import *


class Hybrid_Image_Classification():

    def __init__(
                self,
                data_directory,
                name=None,
                is_dicom=False,
                table=None,
                image_path_column='IMAGE_PATH',
                image_label_column='IMAGE_LABEL',
                is_path=True,
                mode='RAW',
                wl=None,
                # clinical_features=None,
                balance_class=False,
                balance_class_method='upsample',
                interaction_terms=False,
                normalize=((0,0,0), (1,1,1)),
                batch_size=16,
                num_workers=0,
                sampling=1.0,
                test_percent=0.2,
                valid_percent=0.2,
                custom_resize=False,
                model_arch='resnet50',
                pre_trained=True,
                unfreeze=False,
                type='xgboost',
                cv=True,
                stratified=True,
                num_splits=5,
                parameters={},
                # learning_rate=0.0001,
                # epochs=10,
                # optimizer='Adam',
                # loss_function='CrossEntropyLoss',
                # lr_scheduler=None,
                # custom_nn_classifier=None,
                # loss_function_parameters={},
                # optimizer_parameters={},
                transformations='default',
                extra_transformations=None,
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
        self.interaction_terms=interaction_terms
        self.normalize=normalize
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.sampling=sampling
        self.test_percent=test_percent
        self.valid_percent=valid_percent
        self.custom_resize=custom_resize
        self.model_arch=model_arch
        self.pre_trained=pre_trained
        self.unfreeze=unfreeze
        self.type=type
        self.cv=cv
        self.stratified=stratified
        self.num_splits=num_splits
        self.parameters=parameters
        # self.learning_rate=learning_rate
        # self.epochs=epochs
        # self.optimizer=optimizer
        # self.loss_function=loss_function
        # self.lr_scheduler=lr_scheduler
        # self.custom_nn_classifier=custom_nn_classifier
        # self.loss_function_parameters=loss_function_parameters
        # self.optimizer_parameters=optimizer_parameters
        self.transformations=transformations
        self.extra_transformations=extra_transformations
        self.device=device
        self.name=name
        # self.clinical_features=clinical_features

        if self.type=='nn_classifier':
            log ('Error! In hybrid pipeline, CNNs cannot be trained. CNNs are only used for feature extraction.', gui=gui)
            pass

        if self.name==None:
            self.name = 'hybrid_image_classification_'+datetime.now().strftime("%m%d%Y%H%M%S")+'.pipeline'

        if self.device=='auto': self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.type not in SUPPORTED_CLASSIFIER:
            log('Error! Classifier type not supported.')
            pass

        if 'data_processor' not in self.__dict__.keys(): self.data_processor=Data_Processor(**self.__dict__)

        if 'feature_extractor' not in self.__dict__.keys(): self.feature_extractor=Feature_Extractor(dataloader=self.data_processor.master_dataloader, **self.__dict__)

        if 'extracted_feature_dictionary' not in self.__dict__.keys():
            self.train_feature_extractor=Feature_Extractor(dataloader=self.data_processor.train_dataloader, **self.__dict__)
            self.test_feature_extractor=Feature_Extractor(dataloader=self.data_processor.test_dataloader, **self.__dict__)

        # path_col = self.data_processor.table[self.image_path_column]

        self.clinical_features_names = [ x for x in self.table.columns.tolist() if x not in [self.image_label_column, self.image_path_column]]
        self.clinical_features_table = process_categorical(dataframe=self.table[self.clinical_features_names], image_label_column=self.image_label_column, image_path_column=self.image_path_column)
        # self.clinical_features_table = process_categorical(dataframe=self.table[self.clinical_features], image_label_column=self.image_label_column, image_path_column=self.image_path_column)
        # self.clinical_features_names = [x for x in self.clinical_features_table.columns.tolist() if x not in [self.image_label_column]]
        # self.clinical_features_table.insert(0, self.image_path_column,path_col)
        self.clinical_features_table.insert(0, self.image_path_column,self.data_processor.table[self.image_path_column])

    def info(self):
        info=pd.DataFrame.from_dict(({key:str(value) for key, value in self.__dict__.items()}).items())
        info.columns=['Property', 'Value']
        return info

    def run(self, gui=False):
        log('Starting Image Classification Pipeline', gui=gui)
        set_random_seed(100)
        if self.type!='nn_classifier':
            log('Phase 1: Imaging Feature Extraction.', gui=gui)

            if 'extracted_feature_dictionary' in self.__dict__.keys():
                log('Features Already Extracted. Loading Previously Extracted Features', gui=gui)
            else:
                log('Extracting Training Imaging Features', gui=gui)
                self.train_feature_extractor.run(gui=gui)
                log('Extracting Testing Imaging Features', gui=gui)
                self.test_feature_extractor.run(gui=gui)


            log('Phase 2: Combining Clinical and Imaging Features.', gui=gui)

            train_features_names = self.train_feature_extractor.feature_names + self.clinical_features_names
            train_features = pd.merge(self.train_feature_extractor.feature_table, self.clinical_features_table, on=['IMAGE_PATH', 'IMAGE_PATH'])
            train_features = train_features[[x for x in train_features.columns.tolist() if x not in ['IMAGE_PATH','IMAGE_LABEL']]]

            test_features_names = self.test_feature_extractor.feature_names + self.clinical_features_names
            test_features = pd.merge(self.test_feature_extractor.feature_table, self.clinical_features_table, on=['IMAGE_PATH', 'IMAGE_PATH'])
            test_features = test_features[[x for x in test_features.columns.tolist() if x not in ['IMAGE_PATH','IMAGE_LABEL']]]

            self.extracted_feature_dictionary={
                                                'train':{'features':train_features, 'labels':self.train_feature_extractor.labels_idx, 'features_names': train_features_names},
                                                'test':{'features':test_features, 'labels':self.test_feature_extractor.labels_idx, 'features_names': test_features_names}
                                                }

            log('Phase 3: Classifier Training.', gui=gui)
            log ('Running Classifier Training.', gui=gui)
            self.classifier=Classifier(**self.__dict__, )
            self.classifier.run(gui=gui)
            self.trained_model=self.classifier
            self.train_metrics=self.classifier.train_metrics
            # self.feature_selector=Feature_Selector(type=self.classifier.type, feature_table=self.feature_extractor.feature_table, feature_names=self.feature_extractor.feature_names)
            log ('Classifier Training completed successfully.', gui=gui)

        elif self.type=='nn_classifier':
            log ('Error! In hybrid pipeline, CNNs cannot be trained. CNNs are only used for feature extraction.', gui=gui)
            pass

    def metrics(self, figure_size=(700,350)):
        return show_metrics([self.classifier],  figure_size=figure_size)

    def export(self, output_path):
        try:
            outfile=open(output_path,'wb')
            pickle.dump(self,outfile)
            outfile.close()
            log ('Pipeline exported successfully to '+output_path)
        except:
            log ('Error! Pipeline could not be exported.')
            pass
