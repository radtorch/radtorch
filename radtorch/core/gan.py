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


    Parameters
    ----------

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
        x = self.start_num_channels
        for i in range(self.num_units):
            layers.append(self.deconv_unit(input=self.num_generator_features*x, output=self.num_generator_features*int(x/2), kernel_size=4, stride=2, padding=1, bias=False, batch_norm=True, relu=True))
            x = int(x/2)
        layers.append(self.deconv_unit(input=self.num_generator_features, output=self.num_output_channels, kernel_size=4, stride=2, padding=1, bias=False, batch_norm=False, relu=False))
        layers.append(nn.Tanh())
        return layers

    def forward(self, input):
        fc=torch.nn.Linear(self.noise_size, self.num_generator_features*self.num_units*4*4))
        output=fc(input)
        output=output.view((-1, self.num_generator_features*self.num_units, 4, 4))
        ouptut=self.network(output)
        return output


class DCGAN_Discriminator(nn.Module):

    """

    Description
    -----------


    Parameters
    ----------

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
        layer3=nn.LeakyReLU(True)

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
        return layers

    def forward(self, input):
        output = self.network(input)
        ouptut = output.view(-1, self.num_discriminator_features*self.x*self.kernel_size*self.kernel_size)
        fc = nn.Linear(self.num_discriminator_features*self.x*self.kernel_size*self.kernel_size, 1)
        output = fc(output)
