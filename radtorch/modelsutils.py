import torch, torchvision, datetime, time, pickle, pydicom, os
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from sklearn import metrics
from tqdm import tqdm_notebook as tqdm
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path


from radtorch.dicomutils import dicom_to_pil
from radtorch.datautils import IMG_EXTENSIONS



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






def supported():
    '''
    .. include:: ./documentation/docs/modelutils.md##supported

    '''
    print ('Supported Network Architectures:')
    for i in supported_models:
        print (i)
    print('')
    print ('Supported Image Classification Loss Functions:')
    for i in supported_image_classification_losses:
        print (i)
    print('')
    print ('Supported Optimizers:')
    for i in supported_optimizer:
        print (i)
    print ('')
    print ('Supported non DICOM image file types:')
    for i in IMG_EXTENSIONS:
        print (i)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

def create_model(model_arch, output_classes, mode, pre_trained=True, unfreeze_weights=True):
    '''
    .. include:: ./documentation/docs/modelutils.md##create_model
    '''

    if model_arch not in supported_models:
        print ('Error! Provided model architecture is not supported yet. For complete list of supported models please type radtorch.modelsutils.model_list()')

    else:

        if model_arch == 'vgg16' or model_arch == 'vgg19':
            if model_arch == 'vgg16':
                train_model = torchvision.models.vgg16(pretrained=pre_trained)
            elif model_arch == 'vgg19':
                train_model = torchvision.models.vgg19(pretrained=pre_trained)

            if mode == 'feature_extraction':
                train_model.classifier[6] = Identity()
            else:
                train_model.classifier[6] = nn.Sequential(
                    nn.Linear(in_features=4096, out_features=output_classes, bias=True))


        elif model_arch == 'resnet50' or model_arch == 'resnet101' or model_arch == 'resnet152' or model_arch == 'wide_resnet50_2' or  model_arch == 'wide_resnet101_2':
            if model_arch == 'resnet50':
                train_model = torchvision.models.resnet50(pretrained=pre_trained)
            elif model_arch == 'resnet101':
                train_model = torchvision.models.resnet101(pretrained=pre_trained)
            elif  model_arch == 'resnet152':
                train_model = torchvision.models.resnet152(pretrained=pre_trained)
            elif  model_arch == 'wide_resnet50_2':
                train_model = torchvision.models.wide_resnet50_2(pretrained=pre_trained)
            elif  model_arch == 'wide_resnet101_2':
                train_model = torchvision.models.wide_resnet101_2(pretrained=pre_trained)

            fc_inputs = train_model.fc.in_features
            if mode == 'feature_extraction':
                train_model.fc = Identity()
            else:
                train_model.fc = nn.Sequential(
                  nn.Linear(in_features=2048, out_features=output_classes, bias=True))

        elif model_arch == 'inception_v3':
            train_model = torchvision.models.inception_v3(pretrained=pre_trained)
            if mode == 'feature_extraction':
                train_model.fc = Identity()
            else:
                train_model.fc = nn.Linear(in_features=2048, out_features=output_classes, bias=True)


        for param in train_model.parameters():
            param.requires_grad = unfreeze_weights

        return train_model

def create_loss_function(type):
    '''
    .. include:: ./documentation/docs/modelutils.md##create_loss_function
    '''

    if type not in supported_image_classification_losses:
        print ('Error! Provided loss function is not supported yet. For complete list of supported models please type radtorch.modelsutils.supported_list()')

    else:
        loss_function = loss_dict[type]

        return loss_function

def create_optimizer(traning_model, optimizer_type, learning_rate):
    '''
    .. include:: ./documentation/docs/modelutils.md##create_optimizer
    '''

    if optimizer_type=='Adam':
        optimizer = torch.optim.Adam(traning_model.parameters(), lr=learning_rate)
    elif optimizer_type == 'ASGD':
        optimizer = torch.optim.ASGD(traning_model.parameters(), lr=learning_rate)
    elif optimizer_type == 'RMSprop':
        optimizer = torch.optim.RMSprop(traning_model.parameters(), lr=learning_rate)
    elif optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(traning_model.parameters(), lr=learning_rate)

    return optimizer

def train_model(model, train_data_loader, valid_data_loader, train_data_set, valid_data_set,loss_criterion, optimizer, epochs, device,verbose):
    '''
    .. include:: ./documentation/docs/modelutils.md##train_model
    '''

    start_time = datetime.datetime.now()
    training_metrics = []
    if verbose:
        print ('Starting training at '+ str(start_time))


    model = model.to(device)

    for epoch in tqdm(range(epochs)):
        epoch_start = time.time()

        # Set to training mode
        model.train()

        # Loss and Accuracy within the epoch
        train_loss = 0.0
        train_acc = 0.0

        valid_loss = 0.0
        valid_acc = 0.0

        for i, (inputs, labels, image_paths) in enumerate(train_data_loader):
            # inputs = inputs.float()
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Clean existing gradients
            optimizer.zero_grad()

            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)

            # Compute loss
            loss = loss_criterion(outputs, labels)

            # Backpropagate the gradients
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss.item() * inputs.size(0)

            # Compute the accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += acc.item() * inputs.size(0)

            # print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))


        # Validation - No gradient tracking needed
        with torch.no_grad():

            # Set to evaluation mode
            model.eval()

            # Validation loop
            for j, (inputs, labels, image_paths) in enumerate(valid_data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)

                # Compute loss
                loss = loss_criterion(outputs, labels)

                # Compute the total loss for the batch and add it to valid_loss
                valid_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to valid_acc
                valid_acc += acc.item() * inputs.size(0)

                #print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))

        # Find average training loss and training accuracy
        avg_train_loss = train_loss/len(train_data_set)
        avg_train_acc = train_acc/len(train_data_set)

        # Find average validation loss and training accuracy
        avg_valid_loss = valid_loss/len(valid_data_set)
        avg_valid_acc = valid_acc/len(valid_data_set)

        training_metrics.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        epoch_end = time.time()
        if verbose:
            print("Epoch : {:03d}/{} : [Training: Loss: {:.4f}, Accuracy: {:.4f}%]  [Validation : Loss : {:.4f}, Accuracy: {:.4f}%] [Time: {:.4f}s]".format(epoch, epochs, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))

    end_time = datetime.datetime.now()
    total_training_time = end_time-start_time
    if verbose:
        print ('Total training time = '+ str(total_training_time))

    return model, training_metrics

def model_inference(model, input_image_path, inference_transformations=transforms.Compose([transforms.ToTensor()])):
    '''
    .. include:: ./documentation/docs/modelutils.md##model_inference
    '''

    if input_image_path.endswith('dcm'):
        target_img = dicom_to_pil(input_image_path)
    else:
        target_img = Image.open(test_image_name).convert('RGB')

    target_img_tensor = inference_transformations(target_img)
    # target_img_tensor = target_img_tensor.unsqueeze(1)
    target_img_tensor = target_img_tensor.unsqueeze(0)


    with torch.no_grad():
        model.to('cpu')
        target_img_tensor.to('cpu')

        model.eval()
        out = model(target_img_tensor)
        # ps = torch.exp(out)
        ps=out
        prediction_percentages = (ps.cpu().numpy()[0]).tolist()
        pred = prediction_percentages.index(max(prediction_percentages))
        return (pred, max(prediction_percentages))




# def efficientNetNetwork(modelarchitecture, output_classes, pretrained=True):
#
#     if pretrained:
#         trainingNetwork = EfficientNet.from_pretrained('efficientnet-b'+str(modelarchitecture))
#     else:
#         trainingNetwork = EfficientNet.from_name('efficientnet-b'+str(modelarchitecture))
#
#     in_features = trainingNetwork._fc.in_features
#     trainingNetwork._fc =  nn.Linear(in_features=in_features, out_features=output_classes, bias=True)
#
#                                       )
#     return trainingNetwork



##
