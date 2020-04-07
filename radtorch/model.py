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
from radtorch.settings import *
from radtorch.dicom import *
from radtorch.data import *
from radtorch.dataset import *



class Feature_Extractor():
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self,k,v)
        if self.model_arch=='vgg11': self.model=torchvision.models.vgg11(pretrained=self.pre_trained)
        elif self.model_arch=='vgg13':  self.model=torchvision.models.vgg13(pretrained=self.pre_trained)
        elif self.model_arch=='vgg16':  self.model=torchvision.models.vgg16(pretrained=self.pre_trained)
        elif self.model_arch=='vgg19':  self.model=torchvision.models.vgg19(pretrained=self.pre_trained)
        elif self.model_arch=='vgg11_bn': self.model=torchvision.models.vgg11_bn(pretrained=self.pre_trained)
        elif self.model_arch=='vgg13_bn': self.model=torchvision.models.vgg13_bn(pretrained=self.pre_trained)
        elif self.model_arch=='vgg16_bn': self.model=torchvision.models.vgg16_bn(pretrained=self.pre_trained)
        elif self.model_arch=='vgg19_bn': self.model=torchvision.models.vgg19_bn(pretrained=self.pre_trained)
        elif self.model_arch=='resnet18': self.model=torchvision.models.resnet18(pretrained=self.pre_trained)
        elif self.model_arch=='resnet34': self.model=torchvision.models.resnet34(pretrained=self.pre_trained)
        elif self.model_arch=='resnet50': self.model=torchvision.models.resnet50(pretrained=self.pre_trained)
        elif self.model_arch=='resnet101': self.model=torchvision.models.resnet101(pretrained=self.pre_trained)
        elif self.model_arch=='resnet152': self.model=torchvision.models.resnet152(pretrained=self.pre_trained)
        elif self.model_arch=='wide_resnet50_2': self.model=torchvision.models.wide_resnet50_2(pretrained=self.pre_trained)
        elif self.model_arch=='wide_resnet101_2': self.model=torchvision.models.wide_resnet101_2(pretrained=self.pre_trained)
        elif self.model_arch=='alexnet': self.model=torchvision.models.alexnet(pretrained=self.pre_trained)

        if 'vgg' in self.model_arch or 'alexnet' in self.model_arch: self.model.classifier[6]=torch.nn.Identity()
        elif 'resnet' in self.model_arch: self.model.fc=torch.nn.Identity()

        # if self.output_features: #option to reduce complexity of features supplied to classifier
        #     if 'vgg' in self.model_arch or 'alexnet' in self.model_arch: self.model.classifier[6]=torch.nn.Linear(in_features=model_dict[self.model_arch]['output_features'], out_features=self.output_classes, bias=True)
        #     elif 'resnet' in self.model_arch: self.model.fc=torch.nn.Linear(in_features=model_dict[self.model_arch]['output_features'], out_features=self.output_classes, bias=True)


class Classifier(object):
    def __new__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self,k,v)
        self.model=self.feature_extractor.model
        self.model_arch=self.feature_extractor.model_arch
        self.in_features=model_dict[self.model_arch]['output_features']
        if self.type=='linear_regression':
            if 'vgg' in self.model_arch or 'alexnet' in self.model_arch:self.model.classifier[6]=torch.nn.Linear(in_features=self.in_features, out_features=self.output_classes, bias=True)
            elif 'resnet' in self.model_arch: self.model.fc=torch.nn.Linear(in_features=self.in_features, out_features=self.output_classes, bias=True)
        elif self.type=='logistic_regression':
            if self.output_features:
                if 'vgg' in self.model_arch or 'alexnet' in self.model_arch: self.model.classifier[6]=torch.nn.Sequential(
                                    torch.nn.Linear(in_features=self.in_features, out_features=self.output_features, bias=True),
                                    torch.nn.Linear(in_features=self.output_features, out_features=self.output_classes, bias=True),
                                    torch.nn.LogSoftmax(dim=1))
                elif 'resnet' in self.model_arch: self.model.fc=torch.nn.Sequential(
                                    torch.nn.Linear(in_features=self.in_features, out_features=self.output_features, bias=True),
                                    torch.nn.Linear(in_features=self.output_features, out_features=self.output_classes, bias=True),
                                    torch.nn.LogSoftmax(dim=1))
            else:
                if 'vgg' in self.model_arch or 'alexnet' in self.model_arch: self.model.classifier[6]=torch.nn.Sequential(
                                    torch.nn.Linear(in_features=self.in_features, out_features=self.output_classes, bias=True),
                                    torch.nn.LogSoftmax(dim=1))
                elif 'resnet' in self.model_arch: self.model.fc=torch.nn.Sequential(
                                    torch.nn.Linear(in_features=self.in_features, out_features=self.output_classes, bias=True),
                                    torch.nn.LogSoftmax(dim=1))
        return self.model


class Optimizer():
    def __new__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self,k,v)
        if self.type=='Adam':
            self.optimizer=torch.optim.Adam(self.classifier.parameters(),self.learning_rate)
        if self.type=='ASGD':
            self.optimizer=torch.optim.ASGD(self.classifier.parameters(), self.learning_rate)
        if self.type=='RMSprop':
            self.optimizer=torch.optim.RMSprop(self.classifier.parameters(), self.learning_rate)
        if self.type=='SGD':
            self.optimizer=torch.optim.SGD(self.classifier.parameters(), self.learning_rate)
        return self.optimizer


def create_loss_function(type):
    try:
        loss_function=supported_loss[type]
        return loss_function
    except:
        raise TypeError('Error! Provided loss function is not supported yet. For complete list of supported models please type radtorch.modelsutils.supported_list()')
        pass


def train_model(model, train_data_loader, valid_data_loader, train_data_set, valid_data_set,loss_criterion, optimizer, epochs, device, verbose, lr_scheduler):
    '''
    kwargs = model, train_data_loader, valid_data_loader, train_data_set, valid_data_set,loss_criterion, optimizer, epochs, device,verbose
    .. include:: ./documentation/docs/modelutils.md##train_model
    '''
    set_random_seed(100)
    start_time=datetime.datetime.now()
    training_metrics=[]
    if verbose:
        print ('Starting training at '+ str(start_time))


    model=model.to(device)

    for epoch in tqdm(range(epochs)):
        epoch_start=time.time()

        # Set to training mode
        model.train()

        # Loss and Accuracy within the epoch
        train_loss=0.0
        train_acc=0.0

        valid_loss=0.0
        valid_acc=0.0

        for i, (inputs, labels, image_paths) in enumerate(train_data_loader):
            # inputs=inputs.float()
            inputs=inputs.to(device)
            labels=labels.to(device)

            # Clean existing gradients
            optimizer.zero_grad()

            # Forward pass - compute outputs on input data using the model
            outputs=model(inputs)

            # Compute loss
            loss=loss_criterion(outputs, labels)

            # Backpropagate the gradients
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss.item() * inputs.size(0)

            # Compute the accuracy
            ret, predictions=torch.max(outputs.data, 1)
            correct_counts=predictions.eq(labels.data.view_as(predictions))

            # Convert correct_counts to float and then compute the mean
            acc=torch.mean(correct_counts.type(torch.FloatTensor))

            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += acc.item() * inputs.size(0)

            # print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))


        # Validation - No gradient tracking needed
        with torch.no_grad():

            # Set to evaluation mode
            model.eval()

            # Validation loop
            for j, (inputs, labels, image_paths) in enumerate(valid_data_loader):
                inputs=inputs.to(device)
                labels=labels.to(device)

                # Forward pass - compute outputs on input data using the model
                outputs=model(inputs)

                # Compute loss
                loss=loss_criterion(outputs, labels)

                # Compute the total loss for the batch and add it to valid_loss
                valid_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                ret, predictions=torch.max(outputs.data, 1)
                correct_counts=predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc=torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to valid_acc
                valid_acc += acc.item() * inputs.size(0)

                #print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))

        # Find average training loss and training accuracy
        avg_train_loss=train_loss/len(train_data_set)
        avg_train_acc=train_acc/len(train_data_set)

        # Find average validation loss and training accuracy
        avg_valid_loss=valid_loss/len(valid_data_set)
        avg_valid_acc=valid_acc/len(valid_data_set)

        training_metrics.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        epoch_end=time.time()

        if lr_scheduler:
            lr_scheduler.step(avg_valid_loss)

        if verbose:
            print("Epoch : {:03d}/{} : [Training: Loss: {:.4f}, Accuracy: {:.4f}%]  [Validation : Loss : {:.4f}, Accuracy: {:.4f}%] [Time: {:.4f}s]".format(epoch, epochs, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))

    end_time=datetime.datetime.now()
    total_training_time=end_time-start_time
    if verbose:
        print ('Total training time='+ str(total_training_time))

    return model, training_metrics


def model_inference(model, input_image_path, all_predictions=False, inference_transformations=transforms.Compose([transforms.ToTensor()])):
    '''
    .. include:: ./documentation/docs/modelutils.md##model_inference
    '''

    if input_image_path.endswith('dcm'):
        target_img=dicom_to_pil(input_image_path)
    else:
        target_img=Image.open(input_image_path).convert('RGB')

    target_img_tensor=inference_transformations(target_img)
    target_img_tensor=target_img_tensor.unsqueeze(0)


    with torch.no_grad():
        model.to('cpu')
        target_img_tensor.to('cpu')

        model.eval()

        out=model(target_img_tensor)
        softmax=torch.exp(out).cpu()
        prediction_percentages=softmax.cpu().numpy()[0]
        prediction_percentages=[i*100 for i in prediction_percentages]
        _, final_prediction=torch.max(out, 1)
        prediction_table=pd.DataFrame(list(zip([*range(0, len(prediction_percentages), 1)], prediction_percentages)), columns=['label_idx', 'prediction_percentage'])

    if all_predictions:
        return prediction_table
    else:
        return final_prediction.item(), prediction_percentages[final_prediction.item()]
