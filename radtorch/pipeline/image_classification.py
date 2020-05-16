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

class Image_Classification():

    """
    Description
    -----------
    Complete end-to-end image classification pipeline.

    Parameters
    ----------

    - data_directory (string, required): path to target data directory/folder.

    - is_dicom (bollean, optional): True if images are DICOM. default=False.

    - table (string or pandas dataframe, optional): path to label table csv or name of pandas data table. default=None.

    - image_path_column (string, optional): name of column that has image path/image file name. default='IMAGE_PATH'.

    - image_label_column (string, optional): name of column that has image label. default='IMAGE_LABEL'.

    - is_path (boolean, optional): True if file_path column in table is file path. If False, this assumes that the column contains file names only and will append the data_directory to all files. default=True.

    - mode (string, optional): mode of handling pixel values from DICOM to numpy array. Option={'RAW': raw pixel values, 'HU': converts pixel values to HU using slope and intercept, 'WIN':Applies a certain window/level to HU converted DICOM image, 'MWIN': converts DICOM image to 3 channel HU numpy array with each channel adjusted to certain window/level. default='RAW'.

    - wl (tuple or list of tuples, optional): value of Window/Levelto be used. If mode is set to 'WIN' then wl takes the format (level, window). If mode is set to 'MWIN' then wl takes the format [(level1, window1), (level2, window2), (level3, window3)]. default=None.

    - balance_class (bollean, optional): True to perform oversampling in the train dataset to solve class imbalance. default=False.

    - balance_class_method (string, optional): methodology used to balance classes. Options={'upsample', 'downsample'}. default='upsample'.

    - interaction_terms (boolean, optional): create interaction terms between different features and add them as new features to feature table. default=False.

    - normalize (bolean/False or Tuple, optional): Normalizes all datasets by a specified mean and standard deviation. Since most of the used CNN architectures assumes 3 channel input, this follows the following format ((mean, mean, mean), (std, std, std)). default=((0,0,0), (1,1,1)).

    - batch_size (integer, optional): Batch size for dataloader. defult=16.

    - num_workers (integer, optional): Number of CPU workers for dataloader. default=0.

    - sampling (float, optional): fraction of the whole dataset to be used. default=1.0.

    - test_percent (float, optional): percentage of data for testing.default=0.2.

    - valid_percent (float, optional): percentage of data for validation (ONLY with NN_Classifier) .default=0.2.

    - custom_resize (integer, optional): By default, the data processor resizes the image in dataset into the size expected bu the different CNN architectures. To override this and use a custom resize, set this to desired value. default=False.

    - transformations (list, optional): list of pytorch transformations to be applied to all datasets. By default, the images are resized, channels added up to 3 and greyscaled. default='default'.

    - extra_transformations (list, optional): list of pytorch transformations to be extra added to train dataset specifically. default=None.

    - model_arch (string, required): CNN model architecture that this data will be used for. Used to resize images as detailed above. default='alexnet' .

    - pre_trained (boolean, optional): Initialize with ImageNet pretrained weights or not. default=True.

    - unfreeze (boolean, required): Unfreeze all layers of network for future retraining. default=False.

    - type (string, required): type of classifier. For complete list refer to settings. default='logistic_regression'.

    ** Classifier specific parameters:

    - cv (boolean, required): True for cross validation. default=True.

    - stratified (boolean, required): True for stratified cross validation. default=True.

    - num_splits (integer, required): Number of K-fold cross validation splits. default=5.

    - parameters (dictionary, optional): optional parameters passed to the classifier. Please refer to sci-kit learn documentaion.

    ** NN_Classifier specific parameters:

    - learning_rate (float, required): Learning rate. default=0.0001.

    - epochs (integer, required): training epochs. default=10.

    - optimizer (string, required): neural network optimizer type. Please see radtorch.settings for list of approved optimizers. default='Adam'.

    - optimizer_parameters (dictionary, optional): optional extra parameters for optimizer as per pytorch documentation.

    - loss_function (string, required): neural network loss function. Please see radtorch.settings for list of approved loss functions. default='CrossEntropyLoss'.

    - loss_function_parameters (dictionary, optional): optional extra parameters for loss function as per pytorch documentation.

    - lr_scheduler (string, optional): learning rate scheduler - upcoming soon.

    - custom_nn_classifier (pytorch model, optional): Option to use a custom made neural network classifier that will be added after feature extracted layers. default=None.

    - device (string, optional): device to be used for training. Options{'auto': automatic detection of device type, 'cpu': cpu, 'cuda': gpu}. default='auto'.


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
                interaction_terms=False,
                normalize=((0,0,0), (1,1,1)),
                batch_size=16,
                num_workers=0,
                sampling=1.0,
                test_percent=0.2,
                valid_percent=0.2,
                custom_resize=False,
                model_arch='alexnet',
                pre_trained=True,
                unfreeze=False,
                type='nn_classifier',
                cv=True,
                stratified=True,
                num_splits=5,
                parameters={},
                learning_rate=0.0001,
                epochs=10,
                optimizer='Adam',
                loss_function='CrossEntropyLoss',
                lr_scheduler=None,
                custom_nn_classifier=None,
                loss_function_parameters={},
                optimizer_parameters={},
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
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.optimizer=optimizer
        self.loss_function=loss_function
        self.lr_scheduler=lr_scheduler
        self.custom_nn_classifier=custom_nn_classifier
        self.loss_function_parameters=loss_function_parameters
        self.optimizer_parameters=optimizer_parameters
        self.transformations=transformations
        self.extra_transformations=extra_transformations
        self.device=device

        if self.device=='auto': self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if 'data_processor' not in self.__dict__.keys(): self.data_processor=Data_Processor(**self.__dict__)
        if 'feature_extractor' not in self.__dict__.keys(): self.feature_extractor=Feature_Extractor(dataloader=self.data_processor.master_dataloader, **self.__dict__)
        if 'extracted_feature_dictionary' not in self.__dict__.keys():
            self.train_feature_extractor=Feature_Extractor(dataloader=self.data_processor.train_dataloader, **self.__dict__)
            self.test_feature_extractor=Feature_Extractor(dataloader=self.data_processor.test_dataloader, **self.__dict__)

    def info(self):
        info=pd.DataFrame.from_dict(({key:str(value) for key, value in self.__dict__.items()}).items())
        info.columns=['Property', 'Value']
        return info

    def run(self, **kw):
        log('Starting Image Classification Pipeline')
        set_random_seed(100)
        if self.type!='nn_classifier':
            log('Phase 1: Feature Extraction.')

            if 'extracted_feature_dictionary' in self.__dict__.keys():
                log('Features Already Extracted. Loading Previously Extracted Features')
            else:
                log('Extracting Training Features')
                self.train_feature_extractor.run()
                log('Extracting Testing Features')
                self.test_feature_extractor.run()
                self.extracted_feature_dictionary={
                                                    'train':{'features':self.train_feature_extractor.features, 'labels':self.train_feature_extractor.labels_idx, 'features_names': self.train_feature_extractor.feature_names,},
                                                    'test':{'features':self.test_feature_extractor.features, 'labels':self.test_feature_extractor.labels_idx, 'features_names': self.test_feature_extractor.feature_names,}
                                                    }

            log('Phase 2: Classifier Training.')
            log ('Running Classifier Training.')
            self.classifier=Classifier(**self.__dict__, )
            self.classifier.run()
            self.trained_model=self.classifier
            self.train_metrics=self.classifier.train_metrics
            # self.feature_selector=Feature_Selector(type=self.classifier.type, feature_table=self.feature_extractor.feature_table, feature_names=self.feature_extractor.feature_names)
            log ('Classifier Training completed successfully.')
        else:
            self.classifier=NN_Classifier(**self.__dict__)
            self.trained_model, self.train_metrics=self.classifier.run()
            log ('Classifier Training completed successfully.')

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

    def view_score_cam(self, image_path, target_layer, figure_size=(15,5), cmap='rainbow'):
        image=Image.open(image_path).convert('RGB')
        prep_img=self.data_processor.transformations(image)
        prep_img=prep_img.unsqueeze(0)
        prep_img = prep_img.to(self.device)
        cams={}
        score_cam = ScoreCam(self.classifier.trained_model.to(self.device), target_layer=target_layer, image_size=self.data_processor.resize)
        for class_name, class_idx in self.data_processor.classes().items():
            cams[class_name] = score_cam.generate_cam(prep_img, class_idx)
        output_image=prep_img.squeeze(0).squeeze(0).cpu().numpy()
        output_image=np.moveaxis(output_image, 0, -1)

        num_columns = len(cams)+1

        plt.figure(figsize=figure_size)

        plt.subplot(1, num_columns, 1)
        plt.axis('off')
        plt.gca().set_title('Image')
        plt.imshow(output_image, cmap=plt.cm.gray)

        z = 2
        for class_name, cam in cams.items():
            plt.subplot(1, num_columns, z)
            plt.axis('off')
            plt.gca().set_title(class_name)
            plt.imshow(cam, cmap=cmap, alpha=1)
            z = z+1

        plt.show()
