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




supported_models = ['vgg16']
supported_losses = {'NLLLoss':torch.nn.NLLLoss(), 'CrossEntropyLoss':torch.nn.CrossEntropyLoss()}


def supported_list():
    '''
    Returns a list of the currently supported network architectures and loss functions.
    '''
    print ('Supported Network Architectures:')
    for i in supported_models:
        print (i)
    print('')
    print ('Supported Loss Functions:')
    for key, value in supported_losses.items():
        print (key)


def create_model(model_arch, input_channels, output_classes, pre_trained=True):
    '''
    Creates a PyTorch training neural network model with specified network architecture. Input channels and output classes can be specified.
    Inputs:
        model_arch: [str] The architecture of the model neural network. Examples include 'vgg16', 'resnet50', and 'resnet152'.
        pre_trained: [boolen] Load the pretrained weights of the neural network.(default=True)
        input_channels: [int] Number of input image channels. Grayscale DICOM images usually have 1 channel. Colored images have 3.
        output_classes: [int] Number of output classes for image classification problems.
    Outputs:
        Output: [PyTorch neural network object]
    Examples:
        ```
        my_model = radtorch.model.create_model(model_arch='vgg16',input_channels=1, output_classes=2, pre_trained=True)
        ```
    '''

    if model_arch not in supported_models:
        print ('Error! Provided model architecture is not supported yet. For complete list of supported models please type radtorch.modelsutils.model_list()')

    else:

        if model_arch == 'vgg16':
           train_model = torchvision.models.vgg16(pretrained=pre_trained)
           train_model.features[0] = nn.Conv2d(input_channels,64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
           train_model.classifier[6] = nn.Sequential(
                    nn.Linear(in_features=4096, out_features=output_classes, bias=True))



        return train_model

def create_loss_function(type):
    '''
    Creates a PyTorch training loss function object.
    Inputs:
        type: [str] type of the loss functions required.
    Outputs:
        Output: [PyTorch loss function object]
    '''

    if type not in supported_losses:
        print ('Error! Provided loss function is not supported yet. For complete list of supported models please type radtorch.modelsutils.supported_list()')

    else:
        loss_function = supported_losses[type]

        return loss_function

def train_model(model, train_data_loader, valid_data_loader, train_data_set, valid_data_set,loss_criterion, optimizer, epochs, device):
    '''
    Trains a Neural Network Model
    Inputs:
        model: [PyTorch neural network object] Model to be trained.
        train_data_loader: [PyTorch dataloader object] training data dataloader.
        valid_data_loader: [PyTorch dataloader object] validation data dataloader.
        train_data_loader: [PyTorch dataset object] training data dataset.
        valid_data_loader: [PyTorch dataset object] validation data dataset.
        loss_criterion: [PyTorch nn object] Loss function to be used during training.
        optimizer: [PyTorch optimizer object] Optimizer to be used during training.
        epochs: [int] training epochs.
        device: [str] device to be used for training (default='cpu'). This can be 'cpu' or 'gpu'.
    Outputs:
        model: [PyTorch neural network object] trained model.
        train_metrics: [list] list of np arrays of training loss and accuracy.
    Examples:
    ```
    ```
    '''

    start_time = datetime.datetime.now()
    training_metrics = []

    print ('Starting training at '+ str(start_time))


    model = model.to(device)

    for epoch in range(epochs):
        epoch_start = time.time()

        # Set to training mode
        model.train()

        # Loss and Accuracy within the epoch
        train_loss = 0.0
        train_acc = 0.0

        valid_loss = 0.0
        valid_acc = 0.0

        for i, (inputs, labels) in enumerate(train_data_loader):
            inputs = inputs.float()
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
            for j, (inputs, labels) in enumerate(valid_data_loader):
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

        print("Epoch : {:03d}/{} : [Training: Loss: {:.4f}, Accuracy: {:.4f}%]  [Validation : Loss : {:.4f}, Accuracy: {:.4f}%] [Time: {:.4f}s]".format(epoch, epochs, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))

    end_time = datetime.datetime.now()
    total_training_time = end_time-start_time

    print ('Total training time = '+ str(total_training_time))

    return model, training_metrics

# def train_model()
