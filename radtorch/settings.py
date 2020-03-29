
# visutils
TOOLS = "hover,save,box_zoom,reset,wheel_zoom, box_select"
COLORS = ['#1C1533', '#3C6FAA', '#10D8B8', '#FBD704', '#FF7300','#F82716']*100


#modelsutils

model_dict = {'vgg16':{'name':'vgg16','input_size':244, 'output_features':4096},
              'vgg19':{'name':'vgg19','input_size':244, 'output_features':4096},
              'resnet50':{'name':'resnet50','input_size':244, 'output_features':2048},
              'resnet101':{'name':'resnet101','input_size':244, 'output_features':2048},
              'resnet152':{'name':'resnet152','input_size':244, 'output_features':2048},
              'wide_resnet50_2':{'name':'wide_resnet50_2','input_size':244, 'output_features':2048},
              'wide_resnet101_2':{'name':'wide_resnet101_2','input_size':244, 'output_features':2048},
              'inception_v3':{'name':'inception_v3','input_size':299, 'output_features':2048},
              }

loss_dict = {
            'NLLLoss':torch.nn.NLLLoss(),
            'CrossEntropyLoss':torch.nn.CrossEntropyLoss(),
            'MSELoss':torch.nn.MSELoss(),
            'PoissonNLLLoss': torch.nn.PoissonNLLLoss(),
            'BCELoss': torch.nn.BCELoss(),
            'BCEWithLogitsLoss': torch.nn.BCEWithLogitsLoss(),
            'MultiLabelMarginLoss':torch.nn.MultiLabelMarginLoss(),
            'SoftMarginLoss':torch.nn.SoftMarginLoss(),
            'MultiLabelSoftMarginLoss':torch.nn.MultiLabelSoftMarginLoss(),
            }

supported_models = [x for x in model_dict.keys()]

supported_image_classification_losses = ['NLLLoss', 'CrossEntropyLoss']

supported_multi_label_image_classification_losses = []

supported_optimizer = ['Adam', 'ASGD', 'RMSprop', 'SGD']




#datautils
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
