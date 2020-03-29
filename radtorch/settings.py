"""
RADTOrch settings
"""



# visutils
TOOLS = "hover,save,box_zoom,reset,wheel_zoom, box_select"
COLORS = ['#1C1533', '#3C6FAA', '#10D8B8', '#FBD704', '#FF7300','#F82716']*100


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

supported_image_classification_losses = ['NLLLoss', 'CrossEntropyLoss']

supported_multi_label_image_classification_losses = []

supported_optimizer = ['Adam', 'ASGD', 'RMSprop', 'SGD']


#datautils
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
