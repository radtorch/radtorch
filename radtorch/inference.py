import os, torch, pydicom
import pandas as pd
from torch import topk
import matplotlib.image as mpimg

from .utils import *
from .data import *
from .classifier import *
from .extractor import *



class Inference(): #OK
    '''
    An Inference class creates a predictor object that utilizes a trained model to perform predictions over target image(s).

    Args:
        classifier (ImageClassifier): trained [`ImageClassifier`](../classifier/#radtorch.classifier.ImageClassifier).
        use_best_model (bool): True to use the model with the lowest validation loss.
        transform (list, optional): Albumentations transformations. See [Image Augmentation](https://albumentations.ai/docs/getting_started/image_augmentation/). See below.
        device (str): Device to be used for training. Default: 'auto' which automtically detects GPU presence and uses it for feature extraction. Options: 'auto', 'cuda', 'cpu'.

    !!! warning "Using `transform`"
        By default, the `Inference` class utilizes the transforms specified in the `train` subset used to train the `ImageClassifier`. When this is not available, it will try to utilize transforms of the `valid` subset.
        You can specify specific transforms as needed instead.

    '''
    def __init__(self, classifier, use_best_model=True, transform=False, device='auto'):

        self.classifier = classifier
        self.dataset = self.classifier.dataset

        if transform:
            self.transform = transform
        else:
            try:
                self.transform = self.dataset.transform['test']
            except:
                self.transform = self.dataset.transform['train']

        if use_best_model:
            self.model = self.classifier.best_model
        else:
            self.model = self.classifier.model


        self.device = select_device(device)


    def predict(self, img_path, top_predictions='all',  human=True, display_image=False, cmap='gray'):
        '''
        Performs predictions using `Inference` class

        Args:
            img_path (str): path to target image.
            top_predictions (int or str): number of top predictions to return. Default = 'all' which returns all predictions.
            human (bool): True to display predictions in human readable format.
            display_image (bool): True to display the target image.
            cmap (string): Name of Matplotlib color map to be used. See [Matplotlib colormaps](https://matplotlib.org/stable/tutorials/colors/colormaps.html)

        Returns:
            (list) list of predictions if `human` is set to False.


        '''
        if top_predictions == 'all':
            top = len(self.dataset.classes)
        else:
            top = top_predictions
            assert (top <= len(self.dataset.classes)), 'Number of top predictions is more than number of classes. Please check'
        name, ext = os.path.splitext(img_path)
        img = image_to_tensor(img_path, self.dataset.out_channels, self.transform, self.dataset.WW, self.dataset.WL).unsqueeze(0).to(self.device)



        if self.classifier.type == 'sklearn':
            with torch.no_grad():
                self.feature_extractor = self.classifier.feature_extractors['train']
                self.feature_extractor.model.eval()
                nn_output = self.feature_extractor.model(img)
                img_features = pd.DataFrame(nn_output.cpu().numpy())
                raw_pred = self.classifier.best_model.predict_proba(img_features)

        elif self.classifier.type == 'torch':
            with torch.no_grad():
                self.model.eval()
                nn_output = self.model(img.float())
                raw_pred = torch.nn.functional.softmax(nn_output, dim=1).cpu().numpy()

        predictions = []
        s=0
        for i in raw_pred:
            o = []
            for k, v in self.dataset.class_to_idx.items():
                o.append({'id':s, 'class':k, 'class_id':v, 'prob':i.tolist()[v]})
                o = sorted(o, key = lambda i: i['prob'], reverse=True)[:top]
            s = s+1
            predictions.append(o)


        if display_image:
            if ext != '.dcm':
                plt.grid(False)
                plt.imshow(mpimg.imread(img_path), cmap=cmap)
            else:
                plt.grid(False)
                plt.imshow((pydicom.dcmread(img_path).pixel_array), cmap=cmap);

        if human:
            for class_pred in predictions:
                for i in class_pred:
                    print('class: {:4} [prob: {:.2f}%]'.format(i['class'], i['prob']*100))
        else:
            return predictions
