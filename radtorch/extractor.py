import torch
import pandas as pd
import numpy as np
from copy import deepcopy
from torch.utils.model_zoo import load_url
import seaborn as sns
import matplotlib.pyplot as plt


from .utils import *
from .model import *


class FeatureExtractor(): #OK
    '''
    `Feature Extractor` performs feature extraction from images using a [`pytorch model`](https://pytorch.org/vision/stable/models.html) pretrained on [ImageNet](https://image-net.org/).
    Features can be accessed using below attributes after running the feature extraction process through the [`.run()`](../extractor/#radtorch.extractor.FeatureExtractor.run) method.

    Args:
        model_arch (str): Model architecture to be used for feature extraction.
        dataset (ImageDataset): [`ImageDataset`](../data/#radtorch.data.ImageDataset) to be used for training.
        subset (str): the subset op the dataset to extract features from. Default: 'train'. Options: 'train', 'valid', 'test'.
        device (str): Device to be used for training. Default: 'auto' which automtically detects GPU presence and uses it for feature extraction. Options: 'auto', 'cuda', 'cpu'.

    Attributes:
        loader (pytorch dataloader object): Training [pytorch DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).
        table (pandas dataframe): the table of images to be used for feature extraction.
        model (pytorch neural network): Instance of the pytorch model to be used for feature extraction.
        features (pandas dataframe): table of extracted features.
        feature_table (pandas dataframe): table of extracted features and corresponding uid for each image instance.
        feature_names (list): names of the extracted features.

    Examples:
        ```python
        import radtorch
        import albumentations as A

        ds = radtorch.data.ImageDataset('data/PROTOTYPE/FILE/', transforms={'train': A.Compose([A.Resize(64,64)])})

        ext = radtorch.extractor.FeatureExtractor('vgg16', dataset=ds)
        ext.run()

        ```

    '''
    def __init__(self, model_arch, dataset, subset='train', device='auto'):
        self.dataset = dataset
        self.subset = subset
        self.model_arch = model_arch
        assert model_arch in supported_models, 'Error! Selected model architecture not yet Supported. For list of supported models please use supported_models. Thanks'
        self.device = select_device(device)

        # self.model = self._get_pytorch_model()
        self.model = Model(self.model_arch, dataset.out_channels, len(dataset.classes),  pre_trained=True, unfreeze_all=False, vgg_avgpool=False, vgg_fc=False)
        self.model = self._remove_model_last_layer(model=self.model, model_arch=self.model_arch)
        self.model = self.model.to(self.device)
        self.loader = self.dataset.loaders[self.subset]
        self.table = self.dataset.tables[self.subset]


    def run(self):
        '''
        Runs the feature extraction process
        '''

        with torch.no_grad():
            self.model.eval()
            message(' Starting feature extraction of {:} subset using {:} model architecture'.format(self.subset, self.model_arch))
            for i, (images, labels, uid) in tqdm(enumerate(self.loader), total=len(self.loader)):
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model(images.float())
                if i == 0:
                    features, uid_list = deepcopy(output), deepcopy(uid)
                else:
                    features, uid_list = torch.cat((features,output), 0), torch.cat((uid_list, uid), 0)
        self.feature_table = pd.DataFrame(features.cpu().numpy())
        self.feature_table['uid'] = uid_list.numpy()
        self.feature_names = self.feature_table.columns.tolist()[:-1]
        self.features, self.labels = self.hybrid_table(sklearn_ready=True, label_id=True)
        message(' Feature extraction completed successfully.')

    def num_features(self):
        '''
        Returns the expected number of features to be extracted.
        '''
        return self.feature_table.shape[1]

    def _get_pytorch_model(self, pretrained=True):
        model = eval('models.'+self.model_arch+ "()")
        if pretrained:
            state_dict = load_url(model_url[self.model_arch], progress=True)
            model.load_state_dict(state_dict)
        return model

    def _remove_model_last_layer(self, model, model_arch):
        if 'vgg' in model_arch: model.classifier = model.classifier[0]
        elif 'resnet' in model_arch : model.fc = Identity()
        elif 'alexnet' in model_arch: model.classifier = model.classifier[:2]
        elif 'inception' in model_arch: model.fc = Identity()
        return model

    def hybrid_table(self, sklearn_ready=False, label_id=True):
        '''
        Use this method to create pandas dataframes of features and labels that can be used directly into training using scikit-learn.

        Args:
            sklearn_ready (bool): True returns a tuple of extracted features dataframe and labels dataframe. False returns table of features, uid, path, label and label_id.
            label_id (bool): True returns the label ids (integer) instead of the label string.

        '''
        h = pd.merge(self.feature_table, self.table, on='uid')
        h['label_id'] = [self.dataset.class_to_idx[r[self.dataset.label_col]] for i, r in h.iterrows()]
        if sklearn_ready:
            if label_id:
                f, l = self.feature_table[self.feature_names], h['label_id']
            else:
                f, l = self.feature_table[self.feature_names], h[self.dataset.label_col]
            return f, l
        else:
            return h

    def plot_features(self, annotations=False, figure_size=(10,10), cmap="YlGnBu"):
        '''
        Displays a heatmap of the extracted features.

        Args:
            annotations (bool): display values on individual cells.
            figsize (tuple): size of the displayed figure.
            cmap (string): Name of Matplotlib color map to be used. See [Matplotlib colormaps](https://matplotlib.org/stable/tutorials/colors/colormaps.html)

        Returns:
        matplot figure

        '''

        plt.figure(figsize=figure_size)
        plt.title("Extracted Features")
        sns.heatmap(data=self.features, annot=annotations,cmap=cmap);

    def model_info(self):
        batch  = (next(iter(self.loader)))[0]
        batch_size, channels, img_dim, img_dim = batch.shape
        return model_details(self.model, list=False, batch_size=batch_size, channels=channels, img_dim=img_dim)
