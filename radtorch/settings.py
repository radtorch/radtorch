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



"""
RADTOrch settings
"""



# visutils
TOOLS = "hover,save,box_zoom,reset,wheel_zoom, box_select"
COLORS = ['#1C1533', '#3C6FAA', '#10D8B8', '#FBD704', '#FF7300','#F82716',
          '#FF7300', '#FBD704', '#10D8B8', '#3C6FAA', '#1C1533',
          '#3C6FAA', '#10D8B8', '#FBD704', '#FF7300','#F82716',
          '#FF7300', '#FBD704', '#10D8B8', '#3C6FAA', '#1C1533',
          '#3C6FAA', '#10D8B8', '#FBD704', '#FF7300','#F82716',
          '#FF7300', '#FBD704', '#10D8B8', '#3C6FAA', '#1C1533']*100


#modelsutils
model_dict = {'vgg11':{'name':'vgg11','input_size':224, 'output_features':4096},
              'vgg11_bn':{'name':'vgg11_bn','input_size':224, 'output_features':4096},
              'vgg13':{'name':'vgg13','input_size':224, 'output_features':4096},
              'vgg13_bn':{'name':'vgg13_bn','input_size':224, 'output_features':4096},
              'vgg16':{'name':'vgg16','input_size':224, 'output_features':4096},
              'vgg16_bn':{'name':'vgg16_bn','input_size':224, 'output_features':4096},
              'vgg19':{'name':'vgg19','input_size':244, 'output_features':4096},
              'vgg19_bn':{'name':'vgg19_bn','input_size':224, 'output_features':4096},
              'resnet18':{'name':'resnet18','input_size':224, 'output_features':512},
              'resnet34':{'name':'resnet34','input_size':224, 'output_features':512},
              'resnet50':{'name':'resnet50','input_size':224, 'output_features':2048},
              'resnet101':{'name':'resnet101','input_size':224, 'output_features':2048},
              'resnet152':{'name':'resnet152','input_size':224, 'output_features':2048},
              'wide_resnet50_2':{'name':'wide_resnet50_2','input_size':224, 'output_features':2048},
              'wide_resnet101_2':{'name':'wide_resnet101_2','input_size':224, 'output_features':2048},
              # 'inception_v3':{'name':'inception_v3','input_size':299, 'output_features':2048},
              'alexnet':{'name':'alexnet','input_size':256, 'output_features':4096},
              }

supported_models = [x for x in model_dict.keys()]

supported_image_classification_losses = ['NLLLoss', 'CrossEntropyLoss', 'CosineSimilarity']

supported_multi_label_image_classification_losses = []

supported_optimizer = ['Adam', 'ASGD', 'RMSprop', 'SGD']


#datautils
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

#pipeline
IMAGE_CLASSIFICATION_PIPELINE_SETTINGS = {
            'device':'default',
            'optimizer':'Adam',
            'is_dicom': True,
            'label_from_table': False,
            'is_csv': None,
            'table_source': None,
            'path_col':'IMAGE_PATH',
            'label_col' : 'IMAGE_LABEL' ,
            'balance_class' : False,
            'load_predefined_datatables' : False,
            'mode' : 'RAW',
            'wl' : None,
            'normalize' : [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
            'batch_size' : 16,
            'test_percent' : 0.2,
            'valid_percent' : 0.2,
            'model_arch' : 'vgg16',
            'pre_trained' : True,
            'unfreeze_weights' : False,
            'train_epochs' : 20,
            'learning_rate' : 0.0001,
            'loss_function' : 'CrossEntropyLoss',
            'num_workers' : 0,
            'multi_label':False
            }
