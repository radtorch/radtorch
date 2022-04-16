import torch, torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F


from .utils import *


class Model(): #OK
    """
    `Model` class wraps around [`pytorch models`](https://pytorch.org/vision/stable/models.html) to return a new modified version of that model. Modifications include changing the number of input channels, output channels or unfreezing the whole neural network.

    Args:
        model_arch (str): Target model architecture. See [`pytorch models`](https://pytorch.org/vision/stable/models.html).
        in_channels (int): Input Channels.
        out_classes (int): Output Classes.
        pre_trained (bool): If True, returns a model pre-trained on ImageNet.
        unfreeze_all (bool): If True, unfreezes all model weights for training, not just the last FC layer.
        vgg_avgpool (bool): When using VGG model, if True: this changes the `avg_pool` layer to use size (1,1) instead of the default value. Set this to True if the `vgg_fc` is True.
        vgg_fc (bool): When using VGG model, if True: the `classifier` part of the model will be replaced by a single `nn.Linear` layer.

    Returns:
        model: Instance of the target pytorch model with desired modifications.



    !!! danger "Models with 1 input channels"
        Since most of pre-trained models were trained using ImageNet 3-channel images (i.e. RGB), changing the number of input channels to 1 will replace the first [`Conv2d`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) layer in the model with a new layer.

        The new layer weights will be initiallized as noted [here](https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073/2).

        The new layer wweights will be unfrozen by default during training unless you specify so.


    !!! warning "Output FC layer"
        The output FC (`nn.Linear`) layer will be unfrozen by default for training.

    !!! danger "Using VGG models"
        When using VGG models, `vgg_fc` parameter should be specified. Usually set to its default value `True`.

    Examples:
        ```python
        >>> m = radtorch.model.Model('vgg16', 3, 10)
        >>> m
        VGG(
              (features): Sequential(
                (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  #<<< Number of input channels is 3
                (1): ReLU(inplace=True)
                (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (3): ReLU(inplace=True)
                (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (6): ReLU(inplace=True)
                (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (8): ReLU(inplace=True)
                (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (11): ReLU(inplace=True)
                (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (13): ReLU(inplace=True)
                (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (15): ReLU(inplace=True)
                (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (18): ReLU(inplace=True)
                (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (20): ReLU(inplace=True)
                (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (22): ReLU(inplace=True)
                (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (25): ReLU(inplace=True)
                (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (27): ReLU(inplace=True)
                (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (29): ReLU(inplace=True)
                (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
              )
              (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
              (classifier): Linear(in_features=512, out_features=10, bias=True) #<<< Classifier is one layer with output = 10
            )
        ```

        ```python
        >>> m = radtorch.model.Model('vgg16', 1, 10)
        >>> m
        VGG(
              (features): Sequential(
                (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  #<<< Number of input channels is 1
                (1): ReLU(inplace=True)
                (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (3): ReLU(inplace=True)
                (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (6): ReLU(inplace=True)
                (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (8): ReLU(inplace=True)
                (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (11): ReLU(inplace=True)
                (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (13): ReLU(inplace=True)
                (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (15): ReLU(inplace=True)
                (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (18): ReLU(inplace=True)
                (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (20): ReLU(inplace=True)
                (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (22): ReLU(inplace=True)
                (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (25): ReLU(inplace=True)
                (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (27): ReLU(inplace=True)
                (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (29): ReLU(inplace=True)
                (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
              )
              (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
              (classifier): Linear(in_features=512, out_features=10, bias=True) #<<< Classifier is one layer with output = 10
            )
        ```

        ```python
        >>> m = radtorch.model.Model('vgg16', 1, 10, vgg_avgpool=False, vgg_fc=False)
        >>> m
        VGG(
              (features): Sequential(
                (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  #<<< Number of input channels is 3
                (1): ReLU(inplace=True)
                (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (3): ReLU(inplace=True)
                (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (6): ReLU(inplace=True)
                (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (8): ReLU(inplace=True)
                (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (11): ReLU(inplace=True)
                (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (13): ReLU(inplace=True)
                (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (15): ReLU(inplace=True)
                (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (18): ReLU(inplace=True)
                (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (20): ReLU(inplace=True)
                (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (22): ReLU(inplace=True)
                (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (25): ReLU(inplace=True)
                (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (27): ReLU(inplace=True)
                (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (29): ReLU(inplace=True)
                (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
              )
              (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
              (classifier): Sequential(
                (0): Linear(in_features=25088, out_features=4096, bias=True)
                (1): ReLU(inplace=True)
                (2): Dropout(p=0.5, inplace=False)
                (3): Linear(in_features=4096, out_features=4096, bias=True)
                (4): ReLU(inplace=True)
                (5): Dropout(p=0.5, inplace=False)
                (6): Linear(in_features=4096, out_features=10, bias=True) #<<< Output classes = 10
              )
            )
        ```
    """
    def __new__(cls, model_arch, in_channels, out_classes, pre_trained=True, unfreeze_all=False, vgg_avgpool=True, vgg_fc=True, random_seed=0):
        set_random_seed(random_seed)
        return ModelBase(model_arch=model_arch, in_channels=in_channels, out_classes=out_classes, pre_trained=pre_trained, unfreeze_all=unfreeze_all, vgg_avgpool=vgg_avgpool, vgg_fc=vgg_fc).model


class ModelBase(object): #OK
    def __init__(self, *args, **kwargs):
        self.__dict__.update(**kwargs)
        self.model = eval('models.'+self.model_arch+'(pretrained='+str(self.pre_trained)+')')

        assert self.model_arch in supported_models, 'Model not supported yet. See supported_models for details.'

        if 'vgg' in self.model_arch:
            if self.in_channels != 3:
                self.model.features[0] = nn.Conv2d(self.in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.model.classifier[6] = nn.Linear(in_features=4096, out_features=self.out_classes)
            for p in self.model.classifier[6].parameters(): p.requires_grad = True
            if self.vgg_avgpool:
                self.model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
                if self.vgg_fc:
                    self.model.classifier = nn.Linear(in_features=512, out_features=self.out_classes)
            else:
                if self.vgg_fc:
                    self.model.classifier = nn.Linear(in_features=25088, out_features=self.out_classes)

        elif 'resnet' in self.model_arch:
            if self.in_channels != 3:
                self.model.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            if self.model_arch == 'resnet18':
                self.model.fc = nn.Linear(in_features=512, out_features=self.out_classes, bias=True)
            else:
                self.model.fc = nn.Linear(in_features=2048, out_features=self.out_classes, bias=True)
            for p in self.model.fc.parameters(): p.requires_grad = True

        elif 'inception' in self.model_arch:
            if self.in_channels != 3:
                self.model.Conv2d_1a_3x3.conv = nn.Conv2d(self.in_channels, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
            self.model.fc = nn.Linear(in_features=2048, out_features=self.out_classes, bias=True)
            for p in self.model.fc.parameters(): p.requires_grad = True

        elif 'dense' in self.model_arch:
            if in_channels != 3:
                self.model.features.conv0 = nn.Conv2d(self.in_channels, dense_conv0_out_features[self.model_arch[-3:]], kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.model.classifier =  nn.Linear(in_features=dense_classifier_in_features[self.model_arch[-3:]], out_features=self.out_classes, bias=True)

            for p in self.model.classifier.parameters(): p.requires_grad = True

        if self.unfreeze_all:
            for p in self.model.parameters():
                p.requires_grad = True
