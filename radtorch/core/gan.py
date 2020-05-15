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
from .feature_extractor import *



dcgan_generator_options = {
                            16: {'num_units':1, 'start_num_channels':2},
                            32: {'num_units':2, 'start_num_channels':4},
                            64: {'num_units':3, 'start_num_channels':8},
                            128: {'num_units':4, 'start_num_channels':16},
                            256: {'num_units':5, 'start_num_channels':32},
                            512: {'num_units':6, 'start_num_channels':64},
                            1024: {'num_units':7, 'start_num_channels':128},
                            }

dcgan_discriminator_options = {
                            16: {'num_units':1, 'end_num_channels':2},
                            32: {'num_units':2, 'end_num_channels':4},
                            64: {'num_units':3, 'end_num_channels':8},
                            128: {'num_units':4, 'end_num_channels':16},
                            256: {'num_units':5, 'end_num_channels':32},
                            512: {'num_units':6, 'end_num_channels':64},
                            1024: {'num_units':7, 'end_num_channels':128},
                            }


class DCGAN_Generator(nn.Module):

    """

    Description
    -----------
    Core Deep Convolutional GAN Generator Network.


    Parameters
    ----------
    - noise_size (integer, required): size of the noise sample to be generated.

    - num_generator_features (integer, required): number of features/convolutions for generator network.

    - num_output_channels (integer, required): number of channels for output image.

    - target_image_size (integer, required): size of output image.

    - device (string, optional): device to be used for training. Options{'auto': automatic detection of device type, 'cpu': cpu, 'cuda': gpu}. default='auto'.

    """

    def __init__(self, noise_size, num_generator_features, num_output_channels, target_image_size, device='auto'):
        super(DCGAN_Generator, self).__init__()
        self.noise_size=noise_size
        self.num_generator_features=num_generator_features
        self.num_output_channels=num_output_channels
        self.target_image_size=target_image_size
        self.device=device
        if self.device=='auto': self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_units=dcgan_generator_options[target_image_size]['num_units']
        self.start_num_channels=dcgan_generator_options[target_image_size]['start_num_channels']

        self.network = nn.Sequential(*self.network_layers())

    def deconv_unit(self,input, output, kernel_size, stride, padding, bias, batch_norm, relu):
        layer1=nn.ConvTranspose2d(input, output, kernel_size=(kernel_size, kernel_size), stride=(stride,stride), padding=(padding, padding), bias=bias)
        layer2=nn.BatchNorm2d(output)
        layer3=nn.ReLU(True)

        if batch_norm:
            if relu: return nn.Sequential(*[layer1, layer2, layer3])
            else: return nn.Sequential(*[layer1, layer2])
        else:
            if relu: return nn.Sequential(*[layer1, layer3])
            else:  return nn.Sequential(*[layer1])

    def network_layers(self):
        layers=[]
        layers.append(self.deconv_unit(input=self.noise_size, output=self.num_generator_features*self.start_num_channels, kernel_size=4, stride=2, padding=0, bias=False, batch_norm=True, relu=True))
        x = self.start_num_channels
        for i in range(self.num_units):
            layers.append(self.deconv_unit(input=self.num_generator_features*x, output=self.num_generator_features*int(x/2), kernel_size=4, stride=2, padding=1, bias=False, batch_norm=True, relu=True))
            x = int(x/2)
        layers.append(self.deconv_unit(input=self.num_generator_features, output=self.num_output_channels, kernel_size=4, stride=2, padding=1, bias=False, batch_norm=False, relu=False))
        layers.append(nn.Tanh())
        return layers

    def forward(self, input):
        return self.network(input)

    def summary(self):
        summary(self.model, (1, self.noise_size), device=self.device)


class DCGAN_Discriminator(nn.Module):

    """

    Description
    -----------
    Core Deep Convolutional GAN Discriminator Network.


    Parameters
    ----------

    - kernel_size (integer, required): size of kernel/filter to be used for convolution.

    - num_discriminator_features (integer, required): number of features/convolutions for discriminator network.

    - num_input_channels (integer, required): number of channels for input image.

    - input_image_size (integer, required): size of input image.

    - device (string, optional): device to be used for training. Options{'auto': automatic detection of device type, 'cpu': cpu, 'cuda': gpu}. default='auto'.

    """

    def __init__(self, num_input_channels, kernel_size, num_discriminator_features, input_image_size, device='auto'):
        super(DCGAN_Discriminator, self).__init__()
        self.num_input_channels=num_input_channels
        self.num_discriminator_features=num_discriminator_features
        self.input_image_size=input_image_size
        self.device=device
        if self.device=='auto': self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_units=dcgan_discriminator_options[input_image_size]['num_units']
        self.end_num_channels=dcgan_discriminator_options[input_image_size]['end_num_channels']
        self.kernel_size=kernel_size

        self.network = nn.Sequential(*self.network_layers())

    def conv_unit(self,input, output, kernel_size, stride, padding, bias, batch_norm, relu):
        layer1=nn.Conv2d(input, output, kernel_size=(kernel_size, kernel_size), stride=(stride, stride), padding=(padding,padding), bias=bias)
        layer2=nn.BatchNorm2d(output)
        layer3=nn.LeakyReLU(0.2, inplace=True)

        if batch_norm:
            if relu:return nn.Sequential(*[layer1, layer2, layer3])
            else: return nn.Sequential(*[layer1, layer2])
        else:
            if relu: return nn.Sequential(*[layer1, layer3])
            else: return nn.Sequential(*[layer1])


    def network_layers(self):
        layers=[]
        layers.append(self.conv_unit(input=self.num_input_channels, output=self.num_discriminator_features, kernel_size=self.kernel_size, stride=2, padding=1, bias=False, batch_norm=False, relu=True))
        x=1
        for i in range (self.num_units):
            layers.append(self.conv_unit(input=self.num_discriminator_features*x, output=self.num_discriminator_features*(x*2), kernel_size=self.kernel_size, stride=2, padding=1, bias=False, batch_norm=True, relu=True))
            x=x*2
        self.x=x
        layers.append(self.conv_unit(input=self.num_discriminator_features*self.x, output=1, kernel_size=self.kernel_size, stride=2, padding=0, bias=False, batch_norm=False, relu=False))
        layers.append(nn.Sigmoid())
        return layers

    def forward(self, input):
        return self.network(input)

    def summary(self):
        summary(self.model, (self.num_input_channels, self.input_image_size, self.input_image_size), device=self.device)



class GAN_Generator(nn.Module):

    """

    Description
    -----------
    Core Deep Convolutional GAN Generator Network.


    Parameters
    ----------

    - noise_size (integer, required): size of the noise sample to be generated.

    - num_output_channels (integer, required): number of channels for output image.

    - target_image_size (integer, required): size of output image.

    - device (string, optional): device to be used for training. Options{'auto': automatic detection of device type, 'cpu': cpu, 'cuda': gpu}. default='auto'.

    """

    def __init__(self, noise_size, target_image_size, output_num_channels, device='auto'):
        super(GAN_Generator, self).__init__()
        self.noise_size=noise_size
        self.target_image_size=target_image_size
        self.output_num_channels=output_num_channels
        self.device=device
        if self.device=='auto': self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.network = nn.Sequential(*self.network_layers())

    def decoder_unit(self,input, output, batch_norm, relu):
        layer1=nn.Linear(input, output)
        layer2=nn.BatchNorm1d(output)
        layer3=nn.LeakyReLU(0.2, inplace=True)

        if batch_norm:
            if relu: return nn.Sequential(*[layer1, layer2, layer3])
            else: return nn.Sequential(*[layer1, layer2])
        else:
            if relu: return nn.Sequential(*[layer1, layer3])
            else:  return nn.Sequential(*[layer1])

    def network_layers(self):
        output_size=self.target_image_size*self.target_image_size*self.output_num_channels
        layers=[]
        layers.append(self.decoder_unit(input=self.noise_size, output=256, relu=True, batch_norm=True))
        layers.append(self.decoder_unit(input=256, output=512, relu=True, batch_norm=True))
        layers.append(self.decoder_unit(input=512, output=1024, relu=True, batch_norm=True))
        layers.append(self.decoder_unit(input=1024, output=output_size, relu=False, batch_norm=False))
        layers.append(nn.Tanh())
        return layers

    def forward(self, input):
        output = self.network(input)
        output = output.view(-1, self.output_num_channels ,self.target_image_size, self.target_image_size)
        return output

    def summary(self):
        summary(self.model, (1, self.noise_size), device=self.device)

class GAN_Discriminator(nn.Module):

    """

    Description
    -----------
    Core Vanilla GAN Discriminator Network.


    Parameters
    ----------

    - num_input_channels (integer, required): number of channels for input image.

    - input_image_size (integer, required): size of input image.

    - device (string, optional): device to be used for training. Options{'auto': automatic detection of device type, 'cpu': cpu, 'cuda': gpu}. default='auto'.

    """

    def __init__(self, input_image_size, intput_num_channels, device='auto'):
        super(GAN_Discriminator, self).__init__()
        self.input_image_size=input_image_size
        self.intput_num_channels=intput_num_channels
        self.device=device
        if self.device=='auto': self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.network = nn.Sequential(*self.network_layers())

    def encoder_unit(self,input, output, batch_norm, relu):
        layer1=nn.Linear(input, output)
        layer2=nn.BatchNorm1d(output)
        layer3=nn.LeakyReLU(0.2, inplace=True)

        if batch_norm:
            if relu: return nn.Sequential(*[layer1, layer2, layer3])
            else: return nn.Sequential(*[layer1, layer2])
        else:
            if relu: return nn.Sequential(*[layer1, layer3])
            else:  return nn.Sequential(*[layer1])

    def network_layers(self):
        input_size=self.intput_num_channels*self.input_image_size*self.input_image_size
        layers=[]
        layers.append(self.encoder_unit(input=input_size, output=1024, relu=True, batch_norm=True))
        layers.append(self.encoder_unit(input=1024, output=512, relu=True, batch_norm=True))
        layers.append(self.encoder_unit(input=512, output=256, relu=True, batch_norm=True))
        layers.append(self.encoder_unit(input=256, output=1, relu=False, batch_norm=False))
        layers.append(nn.Sigmoid())
        return layers

    def forward(self, input):
        output = input.view(self.intput_num_channels ,self.input_image_size, -1)
        output = self.network(output)
        return output

    def summary(self):
        summary(self.model, (self.num_input_channels, self.input_image_size, self.input_image_size), device=self.device)
