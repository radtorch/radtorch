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



class NN_Classifier():

    """
    Description
    ------------
    Neural Network Classifier. This serves as extension of pytorch neural network modules e.g. VGG16, for fine tuning or transfer learning.


    Parameters
    ----------

    - data_processor (radtorch.core.data_processor, required): data processor object from radtorch.core.Data_Processor.

    - feature_extractor (radtorch.core.feature_extractor, required): feature_extractor object from radtorch.core.Feature_Extractor.

    - unfreeze (boolean, optional): True to unfreeze the weights of all layers in the neural network model for model finetuning. False to just use unfreezed final layers for transfer learning. default=False.

    - learning_rate (float, required): Learning rate. default=0.0001.

    - epochs (integer, required): training epochs. default=10.

    - optimizer (string, required): neural network optimizer type. Please see radtorch.settings for list of approved optimizers. default='Adam'.

    - optimizer_parameters (dictionary, optional): optional extra parameters for optimizer as per pytorch documentation.

    - loss_function (string, required): neural network loss function. Please see radtorch.settings for list of approved loss functions. default='CrossEntropyLoss'.

    - loss_function_parameters (dictionary, optional): optional extra parameters for loss function as per pytorch documentation.

    - lr_scheduler (string, optional): learning rate scheduler - upcoming soon.

    - batch_size (integer, required): batch size. default=16

    - custom_nn_classifier (pytorch model, optional): Option to use a custom made neural network classifier that will be added after feature extracted layers. default=None.

    - device (string, optional): device to be used for training. Options{'auto': automatic detection of device type, 'cpu': cpu, 'cuda': gpu}. default='auto'.

    -

    """

    def __init__(self,
                feature_extractor,
                data_processor,
                unfreeze=False,
                learning_rate=0.0001,
                epochs=10,
                optimizer='Adam',
                loss_function='CrossEntropyLoss',
                lr_scheduler=None,
                batch_size=16,
                device='auto',
                custom_nn_classifier=None,
                loss_function_parameters={},
                optimizer_parameters={},
                **kwargs):

        self.classifier_type='nn_classifier'
        self.type='nn_classifier'
        self.feature_extractor=feature_extractor
        self.data_processor=data_processor
        self.unfreeze=unfreeze
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.optimizer=optimizer
        self.loss_function=loss_function
        self.lr_scheduler=lr_scheduler
        self.batch_size=batch_size
        self.device=device
        self.custom_nn_classifier=custom_nn_classifier
        self.loss_function_parameters=loss_function_parameters
        self.optimizer_parameters=optimizer_parameters

        if self.device=='auto': self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.feature_extractor== None or self.data_processor== None:
            log('Error! No  Data Processor and/or Feature Selector was supplied. Please Check.')
            pass

        # DATA
        self.output_classes=self.data_processor.num_output_classes
        self.train_dataset=self.data_processor.train_dataset
        self.train_dataloader=self.data_processor.train_dataloader
        self.valid_dataset=self.data_processor.valid_dataset
        self.valid_dataloader=self.data_processor.valid_dataloader
        if self.data_processor.test_percent>0:
            self.test_dataset=self.data_processor.test_dataset
            self.test_dataloader=self.data_processor.test_dataloader
        self.transformations=self.data_processor.transformations


        # MODEL
        self.model_arch=self.feature_extractor.model_arch
        self.pre_trained-self.feature_extractor.pre_trained
        if 'efficientnet' in self.model_arch:
            if self.pre_trained:
                self.model=EfficientNet.from_pretrained(self.model_arch, num_classes=self.output_classes)
            else:
                self.model=EfficientNet.from_name(self.model_arch, num_classes=self.output_classes)
        else:
            self.model=copy.deepcopy(self.feature_extractor.model)
        self.in_features=model_dict[self.model_arch]['output_features']

        if self.custom_nn_classifier !=None:
            if 'vgg' in self.model_arch or 'alexnet' in self.model_arch: self.model.classifier=self.custom_nn_classifier
            elif 'resnet' in self.model_arch: self.model.fc=self.custom_nn_classifier
            elif 'efficientnet' in self.model_arch:
                log ('Error! Custom NN_Classifier is not yet supported with EfficientNet.')
                pass

        else:
            if 'vgg' in self.model_arch:
                self.model.classifier=torch.nn.Sequential(
                                torch.nn.Dropout(p=0.5),
                                torch.nn.Linear(in_features=25088, out_features=4096, bias=True),
                                torch.nn.ReLU(inplace=True),
                                torch.nn.Dropout(p=0.5),
                                torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
                                torch.nn.ReLU(inplace=True),
                                torch.nn.Linear(in_features=4096, out_features=self.output_classes, bias=True),
                                torch.nn.LogSoftmax(dim=1))

            elif 'alexnet' in self.model_arch:
                self.model.classifier=torch.nn.Sequential(
                                torch.nn.Dropout(p=0.5),
                                torch.nn.Linear(in_features=9216, out_features=4096, bias=True),
                                torch.nn.ReLU(inplace=True),
                                torch.nn.Dropout(p=0.5),
                                torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
                                torch.nn.ReLU(inplace=True),
                                torch.nn.Linear(in_features=4096, out_features=self.output_classes, bias=True),
                                torch.nn.LogSoftmax(dim=1))


            elif 'resnet' in self.model_arch:
                self.model.fc=torch.nn.Sequential(
                                torch.nn.Linear(in_features=self.in_features, out_features=self.output_classes, bias=True),
                                torch.nn.LogSoftmax(dim=1))

        if self.unfreeze: # This will result in unfreezing and retrain all model layers weights again.
            for param in self.model.parameters():
                param.requires_grad = False




        # Optimizer and Loss Function
        self.loss_function=self.nn_loss_function(type=self.loss_function, **self.loss_function_parameters)
        self.optimizer=self.nn_optimizer(type=self.optimizer, model=self.model, learning_rate=self.learning_rate,  **self.optimizer_parameters)

    def info(self):

        """
        Returns table with all information about the nn_classifier object.

        """

        info=pd.DataFrame.from_dict(({key:str(value) for key, value in self.__dict__.items()}).items())
        info.columns=['Property', 'Value']
        for i in ['train_dataset', 'valid_dataset','test_dataset']:
            if i in self.__dict__.keys():
                info.append({'Property':i+' size', 'Value':len(self.__dict__[i])}, ignore_index=True)
        return info

    def nn_optimizer(self, type, model, learning_rate, **kw):

        """

        Description
        -----------
        Creates an instance of pytorch optimizer


        Parameters
        ----------

        - type (string, required): type of the optimizer. Please see settings for supported optimizers.

        - model (pytorch model, required): model for which optimizer will be used for weight optimization.

        - learning_rate (float, required): learning rate for training.

        - **kw (dictionary, optional): other optional optimizer parameters as per pytorch documentation.

        Returns
        -------
        pytorch nn.optimizer object

        """

        if type not in supported_nn_optimizers:
            log('Error! Optimizer not supported yet. Please check radtorch.settings.supported_nn_optimizers')
            pass
        elif type=='Adam':
            optimizer=torch.optim.Adam(params=model.parameters(),lr=learning_rate, **kw)
        elif type=='AdamW':
            optimizer=torch.optim.AdamW(params=model.parameters(), lr=learning_rate, **kw)
        elif type=='SparseAdam':
            optimizer=torch.optim.SparseAdam(params=model.parameters(), lr=learning_rate, **kw)
        elif type=='Adamax':
            optimizer=torch.optim.Adamax(params=model.parameters(), lr=learning_rate, **kw)
        elif type=='ASGD':
            optimizer=torch.optim.ASGD(params=model.parameters(), lr=learning_rate, **kw)
        elif type=='RMSprop':
            optimizer=torch.optim.RMSprop(params=model.parameters(), lr=learning_rate, **kw)
        elif type=='SGD':
            optimizer=torch.optim.SGD(params=model.parameters(), lr=learning_rate, **kw)
        log('Optimizer selected is '+type)
        return optimizer

    def nn_loss_function(self, type, **kw):
        """

        Description
        -----------
        Creates an instance of pytorch loss function.

        Parameters
        ----------

        - type (string, required): type of the loss function. Please see settings for supported loss functions.

        - **kw (dictionary, optional): other optional loss function parameters as per pytorch documentation.

        Returns
        -------
        pytorch nn.loss_function object

        """

        if type not in supported_nn_loss_functions:
            log('Error! Loss functions not supported yet. Please check radtorch.settings.supported_nn_loss_functions')
            pass
        elif type== 'NLLLoss':
            loss_function=torch.nn.NLLLoss(**kw),
        elif type== 'CrossEntropyLoss':
            loss_function=torch.nn.CrossEntropyLoss(**kw)
        elif type== 'MSELoss':
            loss_function=torch.nn.MSELoss(**kw)
        elif type== 'PoissonNLLLoss':
            loss_function=torch.nn.PoissonNLLLoss(**kw)
        elif type== 'BCELoss':
            loss_function=torch.nn.BCELoss(**kw)
        elif type== 'BCEWithLogitsLoss':
            loss_function=torch.nn.BCEWithLogitsLoss(**kw)
        elif type== 'MultiLabelMarginLoss':
            loss_function=torch.nn.MultiLabelMarginLoss(**kw)
        elif type== 'SoftMarginLoss':
            loss_function=torch.nn.SoftMarginLoss(**kw)
        elif type== 'MultiLabelSoftMarginLoss':
            loss_function=torch.nn.MultiLabelSoftMarginLoss(**kw)
        elif type== 'CosineSimilarity':
            loss_function=torch.nn.CosineSimilarity(**kw)
        log('Loss function selected is '+type)
        return loss_function

    def run(self, **kw):
        """
        Performs Model Training

        Returns
        --------
        Tuple of
            - trained_model: trained neural network model.
            - train_metrics: pandas dataframe of training and validation metrics.
        """

        model=self.model
        train_data_loader=self.train_dataloader
        valid_data_loader=self.valid_dataloader
        train_data_set=self.train_dataset
        valid_data_set=self.valid_dataset
        loss_criterion=self.loss_function
        optimizer=self.optimizer
        epochs=self.epochs
        device=self.device
        if self.lr_scheduler!=None: lr_scheduler=self.lr_scheduler
        else: lr_scheduler=False

        set_random_seed(100)
        start_time=datetime.now()
        training_metrics=[]
        if self.unfreeze:
            log('INFO: unfreeze is set to True. This will unfreeze all model layers and will train from scratch. This might take sometime specially if pre_trained=False.')
        log('Starting training at '+ str(start_time))
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
            log("Epoch : {:03d}/{} : [Training: Loss: {:.4f}, Accuracy: {:.4f}%]  [Validation : Loss : {:.4f}, Accuracy: {:.4f}%] [Time: {:.4f}s]".format(epoch, epochs, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))
        end_time=datetime.now()
        total_training_time=end_time-start_time
        log('Total training time='+ str(total_training_time))
        self.trained_model=model
        self.train_metrics=training_metrics
        self.train_metrics = pd.DataFrame(data=self.train_metrics, columns = ['Train_Loss', 'Valid_Loss', 'Train_Accuracy', 'Valid_Accuracy'])
        return self.trained_model, self.train_metrics

    def confusion_matrix(self, target_dataset=None, figure_size=(8,6), cmap=None):

        """
        Displays confusion matrix for trained nn_classifier on test dataset.

        Parameters
        ----------

        - target_dataset (pytorch dataset, optional): this option can be used to test the trained model on an external test dataset. If set to None, the confusion matrix is generated using the test dataset initially specified in the data_processor. default=None.

        - figure_size (tuple, optional): size of the figure as width, height. default=(8,6)

        """

        if target_dataset==None:target_dataset=self.test_dataset
        target_classes=(self.data_processor.classes()).keys()

        show_nn_confusion_matrix(model=self.trained_model, target_data_set=target_dataset, target_classes=target_classes, device=self.device, figure_size=figure_size, cmap=cmap)

    def roc(self, **kw):
        """
        Displays ROC and AUC of trained model with test dataset

        """
        show_roc([self], **kw)

    def metrics(self, figure_size=(700,400)):

        """
        Displays graphical representation of train/validation loss /accuracy.

        Parameters
        ----------

        - figure_size (tuple, optional): size of the figure as width, height. default=(700,400)

        """

        show_metrics([self], figure_size=figure_size)

    def predict(self,  input_image_path, all_predictions=True, **kw):

        """
        Description
        -----------
        Displays classs prediction for a target image using a trained classifier.


        Parameters
        ----------

        - input_image_path (string, required): path to target image.

        - all_predictions (boolean, optional): True to display prediction percentage accuracies for all prediction classes. default=True.

        """

        model=self.trained_model
        transformations=self.transformations

        if input_image_path.endswith('dcm'):
            target_img=dicom_to_pil(input_image_path)
        else:
            target_img=Image.open(input_image_path).convert('RGB')

        target_img_tensor=transformations(target_img)
        target_img_tensor=target_img_tensor.unsqueeze(0)

        with torch.no_grad():
            model.to('cpu')
            target_img_tensor.to('cpu')
            model.eval()
            out=model(target_img_tensor)
            softmax=torch.exp(out).cpu()
            prediction_percentages=softmax.cpu().numpy()[0]
            # prediction_percentages=[i*100 for i in prediction_percentages]
            prediction_percentages = [("%.4f" % x) for x in prediction_percentages]
            _, final_prediction=torch.max(out, 1)
            prediction_table=pd.DataFrame(list(zip(self.data_processor.classes().keys(), [*range(0, len(prediction_percentages), 1)], prediction_percentages)), columns=['LABEL','LABEL_IDX', 'PREDICTION_ACCURACY'])

        if all_predictions:
            return prediction_table
        else:
            return final_prediction.item(), prediction_percentages[final_prediction.item()]

    def misclassified(self, num_of_images=4, figure_size=(5,5), table=False, **kw):

        """
        Description
        -----------
        Displays sample of images misclassified by the classifier from test dataset.


        Parameters
        ----------

        - num_of_images (integer, optional): number of images to be displayed. default=4.

        - figure_size (tuple, optional): size of the figure as width, height. default=(5,5).

        - table (boolean, optional): True to display a table of all misclassified images including image path, true label and predicted label.

        """
        misclassified_table = show_nn_misclassified(model=self.trained_model, target_data_set=self.test_dataset, num_of_images=num_of_images, device=self.device, transforms=self.data_processor.transformations, is_dicom = self.is_dicom, figure_size=figure_size)
        if table:
            return misclassified_table

    def summary(self):
        summary(self.model.to(self.device), (3, model_dict[self.model_arch]['input_size'], model_dict[self.model_arch]['input_size']), device=str(self.device))


    def show_model_layers(self):
        return [x for x, y in self.trained_model.named_modules()]
