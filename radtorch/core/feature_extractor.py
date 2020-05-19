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
from .data_processor import *



class Feature_Extractor():

    """

    Description
    -----------
    Creates a feature extractor neural network using one of the famous CNN architectures and the data provided as dataloader from Data_Processor.

    Parameters
    ----------

    - model_arch (string, required): CNN architecture to be utilized. To see list of supported architectures see settings.

    - pre_trained (boolean, optional): Initialize with ImageNet pretrained weights or not. default=True.

    - unfreeze (boolean, required): Unfreeze all layers of network for future retraining. default=False.

    - dataloader (pytorch dataloader object, required): the dataloader that will be used to supply data for feature extraction.

    - device (string, optional): device to be used for training. Options{'auto': automatic detection of device type, 'cpu': cpu, 'cuda': gpu}. default='auto'.

    Retruns
    ---------
    Pandas dataframe with extracted features.

    """


    def __init__(
                self,
                model_arch,
                dataloader,
                pre_trained=True,
                unfreeze=False,
                device='auto',
                **kwargs):

        self.model_arch=model_arch
        self.dataloader=dataloader
        self.pre_trained=pre_trained
        self.unfreeze=unfreeze
        self.device=device

        for k,v in kwargs.items():
            setattr(self,k,v)
        if self.model_arch not in supported_models:
            log('Error! Provided model architecture is not yet suported. Please use radtorch.settings.supported_models to see full list of supported models.')
            pass
        elif self.model_arch=='vgg11': self.model=torchvision.models.vgg11(pretrained=self.pre_trained)
        elif self.model_arch=='vgg13':  self.model=torchvision.models.vgg13(pretrained=self.pre_trained)
        elif self.model_arch=='vgg16':  self.model=torchvision.models.vgg16(pretrained=self.pre_trained)
        elif self.model_arch=='vgg19':  self.model=torchvision.models.vgg19(pretrained=self.pre_trained)
        elif self.model_arch=='vgg11_bn': self.model=torchvision.models.vgg11_bn(pretrained=self.pre_trained)
        elif self.model_arch=='vgg13_bn': self.model=torchvision.models.vgg13_bn(pretrained=self.pre_trained)
        elif self.model_arch=='vgg16_bn': self.model=torchvision.models.vgg16_bn(pretrained=self.pre_trained)
        elif self.model_arch=='vgg19_bn': self.model=torchvision.models.vgg19_bn(pretrained=self.pre_trained)
        elif self.model_arch=='resnet18': self.model=torchvision.models.resnet18(pretrained=self.pre_trained)
        elif self.model_arch=='resnet34': self.model=torchvision.models.resnet34(pretrained=self.pre_trained)
        elif self.model_arch=='resnet50': self.model=torchvision.models.resnet50(pretrained=self.pre_trained)
        elif self.model_arch=='resnet101': self.model=torchvision.models.resnet101(pretrained=self.pre_trained)
        elif self.model_arch=='resnet152': self.model=torchvision.models.resnet152(pretrained=self.pre_trained)
        elif self.model_arch=='wide_resnet50_2': self.model=torchvision.models.wide_resnet50_2(pretrained=self.pre_trained)
        elif self.model_arch=='wide_resnet101_2': self.model=torchvision.models.wide_resnet101_2(pretrained=self.pre_trained)
        elif self.model_arch=='alexnet': self.model=torchvision.models.alexnet(pretrained=self.pre_trained)
        elif 'efficientnet' in self.model_arch:
            if self.pre_trained:
                self.model=EfficientNet.from_pretrained(self.model_arch)
            else:
                self.model=EfficientNet.from_name(self.model_arch)

        if 'alexnet' in self.model_arch or 'vgg' in self.model_arch:
            self.model.classifier[6]=torch.nn.Identity()

        elif 'resnet' in self.model_arch:
            self.model.fc=torch.nn.Identity()



        if self.unfreeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def num_features(self):
        """
        Returns the number of features to be extracted.
        """
        return model_dict[self.model_arch]['output_features']

    def run(self, verbose=False):
        """
        Runs the feature exraction process

        Returns
        --------
        tuple of feature_table (dataframe which contains all features, labels and image file path), features (dataframe which contains features only), feature_names(list of feature names)

        """

        if 'balance_class' in self.__dict__.keys() and 'normalize' in self.__dict__.keys():
            log('Running Feature Extraction using '+str(self.model_arch)+' architecture with balance_class = '+str(self.balance_class)+' and normalize = '+str(self.normalize)+".")
        else:
            log('Running Feature Extraction using '+str(self.model_arch)+' architecture')
        self.features=[]
        self.labels_idx=[]
        self.img_path_list=[]
        self.model=self.model.to(self.device)
        for i, (imgs, labels, paths) in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
            self.labels_idx=self.labels_idx+labels.tolist()
            self.img_path_list=self.img_path_list+list(paths)
            with torch.no_grad():
                self.model.eval()
                imgs=imgs.to(self.device)
                # if 'efficientnet' in self.model_arch:
                #     output = (self.model.extract_features(imgs)).tolist()
                # else:
                output=(self.model(imgs)).tolist()
                self.features=self.features+(output)
        self.feature_names=['f_'+str(i) for i in range(0,self.num_features())]
        feature_table=pd.DataFrame(list(zip(self.img_path_list, self.labels_idx, self.features)), columns=['IMAGE_PATH','IMAGE_LABEL', 'FEATURES'])
        feature_table[self.feature_names]=pd.DataFrame(feature_table.FEATURES.values.tolist(), index= feature_table.index)
        feature_table=feature_table.drop(['FEATURES'], axis=1)
        log('Features extracted successfully.')
        self.feature_table=feature_table
        self.features=self.feature_table[self.feature_names]
        return self.feature_table, self.features, self.feature_names
        if verbose:
            print (self.feature_table)

    def export_features(self,csv_path):

        """
        Exports extracted features into csv file.

        Parameters
        ----------
        csv_path (string, required): path to csv output.

        """
        try:
            self.feature_table.to_csv(csv_path, index=False)
            log('Features exported to CSV successfully.')
        except:
            log('Error! No features found. Please check again or re-run the extracion pipeline.')
            pass

    def plot_extracted_features(self, num_features=100, num_images=100,image_path_column='IMAGE_PATH', image_label_column='IMAGE_LABEL'):
        """
        Plots Extracted Features in Heatmap

        Parameters
        -----------

        - num_features (integer, optional): number of features to display. default=100

        - num_images (integer, optional): number of images to display features for. default=100

        - image_path_column (string, required): name of column that has image names/path. default='IMAGE_PATH'

        - image_label_column (string, required): name of column that has image labels. default='IMAGE_LABEL'

        """

        return plot_features(feature_table=self.feature_table, feature_names=self.feature_names, num_features=num_features, num_images=num_images,image_path_col=image_path_column, image_label_col=image_label_column)

    def export(self, output_path):
        """
        Exports the Feature Extractor object for future use.

        Parameters
        ----------
        output_path (string, required): output file path.

        """
        try:
            outfile=open(output_path,'wb')
            pickle.dump(self,outfile)
            outfile.close()
            log('Feature Extractor exported successfully.')
        except:
            raise TypeError('Error! Feature Extractor could not be exported.')

    def summary(self):
        summary(self.model, (3, model_dict[self.model_arch]['input_size'], model_dict[self.model_arch]['input_size']), device=str(self.device))
