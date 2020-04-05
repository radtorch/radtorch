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
            'CosineSimilarity':torch.nn.CosineSimilarity(dim=1, eps=1e-08),
            }



def set_device(device):
    if device == 'default':
        selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        selected_device == device
    return selected_device

def set_transformations(model_arch, custom_resize, is_dicom, transformations):
    if custom_resize=='default':
        input_resize = model_dict[model_arch]['input_size']
    else:
        input_resize = custom_resize

    if transformations == 'default':
        if is_dicom == True:
            transformations = transforms.Compose([
                    transforms.Resize((input_resize, input_resize)),
                    transforms.transforms.Grayscale(3),
                    transforms.ToTensor()])
        else:
            transformations = transforms.Compose([
                    transforms.Resize((input_resize, input_resize)),
                    transforms.ToTensor()])

    return transformations, input_resize


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

def create_model(**kwargs):
    '''
    model_arch, output_classes, mode, pre_trained=True, unfreeze_weights=False,
    .. include:: ./documentation/docs/modelutils.md##create_model
    '''

    model_arch = kwargs['model_arch']
    output_classes = kwargs['output_classes']
    mode = kwargs['mode']

    if model_arch not in supported_models:
        print ('Error! Provided model architecture is not supported yet. For complete list of supported models please type radtorch.modelsutils.model_list()')

    else:

        if 'vgg' in model_arch:
            in_features = model_dict[model_arch]['output_features']
            if model_arch == 'vgg11':
                train_model = torchvision.models.vgg11(pretrained=pre_trained)
            if model_arch == 'vgg13':
                train_model = torchvision.models.vgg13(pretrained=pre_trained)
            if model_arch == 'vgg16':
                train_model = torchvision.models.vgg16(pretrained=pre_trained)
            elif model_arch == 'vgg19':
                train_model = torchvision.models.vgg19(pretrained=pre_trained)
            if model_arch == 'vgg11_bn':
                train_model = torchvision.models.vgg11_bn(pretrained=pre_trained)
            if model_arch == 'vgg13_bn':
                train_model = torchvision.models.vgg13_bn(pretrained=pre_trained)
            if model_arch == 'vgg16_bn':
                train_model = torchvision.models.vgg16_bn(pretrained=pre_trained)
            elif model_arch == 'vgg19_bn':
                train_model = torchvision.models.vgg19_bn(pretrained=pre_trained)
            if mode == 'feature_extraction':
                train_model.classifier[6] = Identity()
            elif mode == 'feature_visualization':
                train_model.classifier[6] = nn.Sequential(
                    nn.Linear(in_features=in_features, out_features=output_classes, bias=True))
            else:
                train_model.classifier[6] = nn.Sequential(
                    nn.Linear(in_features=in_features, out_features=output_classes, bias=True),
                    torch.nn.LogSoftmax(dim=1)
                    )



        elif 'resnet' in model_arch:
            in_features = model_dict[model_arch]['output_features']
            if model_arch == 'resnet18':
                train_model = torchvision.models.resnet18(pretrained=pre_trained)
            elif model_arch == 'resnet34':
                train_model = torchvision.models.resnet34(pretrained=pre_trained)
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
            elif mode == 'feature_visualization':
                train_model.fc = nn.Sequential(
                  nn.Linear(in_features=in_features, out_features=output_classes, bias=True))
            else:
                train_model.fc = nn.Sequential(
                  nn.Linear(in_features=in_features, out_features=output_classes, bias=True),
                  torch.nn.LogSoftmax(dim=1)
                  )


        # elif model_arch == 'inception_v3':
        #     train_model = torchvision.models.inception_v3(pretrained=pre_trained)
        #     if mode == 'feature_extraction':
        #         train_model.fc = Identity()
        #     elif mode == 'feature_visualization':
        #         train_model.fc = nn.Sequential(
        #           nn.Linear(in_features=2048, out_features=output_classes, bias=True))
        #     else:
        #         train_model.fc = nn.Sequential(
        #           nn.Linear(in_features=2048, out_features=output_classes, bias=True),
        #           torch.nn.LogSoftmax(dim=1)
        #           )

        elif model_arch == 'alexnet':
            in_features = model_dict[model_arch]['output_features']
            train_model = torchvision.models.alexnet(pretrained=pre_trained)
            if mode == 'feature_extraction':
                train_model.classifier[6] = Identity()
            elif mode == 'feature_visualization':
                train_model.classifier[6] = nn.Sequential(
                  nn.Linear(in_features=in_features, out_features=output_classes, bias=True))
            else:
                train_model.classifier[6] = nn.Sequential(
                  nn.Linear(in_features=in_features, out_features=output_classes, bias=True),
                  torch.nn.LogSoftmax(dim=1)
                  )

        if pre_trained == False:
            for param in train_model.parameters():
                param.requires_grad = unfreeze_weights

        else:
            if unfreeze_weights:
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

def create_optimizer(**kwargs):
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
    set_random_seed(100)
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

def model_inference(model, input_image_path, all_predictions = False, inference_transformations=transforms.Compose([transforms.ToTensor()])):
    '''
    .. include:: ./documentation/docs/modelutils.md##model_inference
    '''

    if input_image_path.endswith('dcm'):
        target_img = dicom_to_pil(input_image_path)
    else:
        target_img = Image.open(input_image_path).convert('RGB')

    target_img_tensor = inference_transformations(target_img)
    target_img_tensor = target_img_tensor.unsqueeze(0)


    with torch.no_grad():
        model.to('cpu')
        target_img_tensor.to('cpu')

        model.eval()

        out = model(target_img_tensor)
        softmax = torch.exp(out).cpu()
        prediction_percentages = softmax.cpu().numpy()[0]
        prediction_percentages = [i*100 for i in prediction_percentages]
        _, final_prediction = torch.max(out, 1)
        prediction_table = pd.DataFrame(list(zip([*range(0, len(prediction_percentages), 1)], prediction_percentages)), columns=['label_idx', 'prediction_percentage'])

    if all_predictions:
        return prediction_table
    else:
        return final_prediction.item(), prediction_percentages[final_prediction.item()]







##
