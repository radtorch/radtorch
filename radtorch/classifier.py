import torch
import pandas as pd
import seaborn as sns

from .utils import *
from .extractor import *



class ImageClassifier():
    """
    Class for image classifier. This class acts as wrapper to train a selected model (either pytorch neural network or a sklearn classifier) using a dataset which can be either a radtorch [`ImageDataset`](../data/#radtorch.data.ImageDataset) or [`VolumeDataset`](../data/#radtorch.data.VolumeDataset).

    Optionally, a specific train and validation pytorch dataloaders may be manually specified instead of using radtorch dataset objects.

    !!! info "Training a Pytroch Neural Network"
        If the model to train is a pytorch neural network, in addition to the model, `ImageClassifier` expects a pytorch [criterion/loss function](https://pytorch.org/docs/stable/nn.html#loss-functions), a [pytorch optimizer](https://pytorch.org/docs/stable/optim.html) and an optional [pytorch scheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate).

    !!! info "Training an sklearn classifier"
        If the model to be trained is an sklearn classifier, `ImageClassifier` performs feature extraction followed by training the model. Accordingly,  `ImageClassifier` expects a model architecture for the feature extraction process.

    !!! danger "Creating multiple classifier objects using same model/neural network object"
        To ensure results consistency,  a new instance of pytorch model/neural network object MUST be instatiated with every classifier object.

        For example, Do this:

        ```python
        M =radtorch.model.Model(model_arch='vgg16', in_channels=1, out_classes=2)
        clf = radtorch.classifier.ImageClassifier(model=M, dataset=ds)
        clf.fit(epochs=3)

        M =radtorch.model.Model(model_arch='vgg16', in_channels=1, out_classes=2)
        clf2 = radtorch.classifier.ImageClassifier(model=M, dataset=ds)
        clf2.fit(epochs=3)
        ```

        and **Do NOT** do this :

        ```python
        M =radtorch.model.Model(model_arch='vgg16', in_channels=1, out_classes=2)

        clf = radtorch.classifier.ImageClassifier(model=M, dataset=ds)
        clf.fit(epochs=3)

        clf2 = radtorch.classifier.ImageClassifier(model=M, dataset=ds)
        clf2.fit(epochs=3)
        ```

    Args:
        name (str, optional): Name to be give to the Image Classifier. If none provided, the current date and time will be used to created a generic classifier name.
        model (pytroch neural network or sklearn classifier): Model to be trained.
        dataset (ImageDataset or VolumeDataset): [`ImageDataset`](../data/#radtorch.data.ImageDataset) or [`VolumeDataset`](../data/#radtorch.data.VolumeDataset) to be used for training.
        dataloader_train (pytorch dataloader, optional): Optional Training [pytorch DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
        dataloader_valid (pytorch dataloader, optional): Optional Validation [pytorch DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
        device (str): Device to be used for training.
        feature_extractor_arch (str, optional): Architecture of the model to be used for feature extraction when training sklearn classifier. See (https://pytorch.org/vision/0.8/models.html#classification)[https://pytorch.org/vision/0.8/models.html#classification]
        criterion (pytorch loss function): Loss function to be used during training a pytorch neural network.
        optimizer (pytorch optimizer): Loss function to be used during training a pytorch neural network.
        scheduler (pytorch scheduler, optional): Scheduler to be used during training a pytorch neural network.
        scheduler metric (str): when using [ReduceLROnPlateau](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau) pytorch scheduler, a target loss or accuracy must be provided to monitor. Options: 'train_loss', 'train_accuracy', 'valid_loss', 'valid_accuracy'.
        use_checkpoint (bool): Path (str) to a saved checkpoint to continue training. If a checkpoint is used to resume training, training will be resumed from saved checkpoint to new/specified epoch number.
        random_seed (int, optional): Random seed (default=100)

    !!! danger "Using manual pytorch dataloaders"
        If maually created dataloaders are used, set `dataset` to None.

    !!! tip "Selecting device for training"
        `Auto` mode automatically detects if there is GPU utilizes it for training.

    Attributes:
        type (str): Type of the classifier model to be trained.
        train_losses (list): List of train losses recorded. Length = Number of epochs.
        valid_losses (list): List of validation losses recorded. Length = Number of epochs.
        train_acc (list): List of train accuracies recorded. Length = Number of epochs.
        valid_acc (list): List of validation accuracies recorded. Length = Number of epochs.
        valid_loss_min (float): Minimum Validation Loss to save checkpoint.
        best_model (pytroch neural network or sklearn classifier): Best trained model with lowest `Validation Loss` in case of pytorch neural networks or the trained classifier for sklearn classifiers.
        train_logs (pandas dataframe): Table/Dataframe with all train/validation losses.

    """
    def __init__(self, name=None, model=None, dataset=None, dataloader_train=None, dataloader_valid=None, device='auto', feature_extractor_arch='vgg16', criterion=None, optimizer=None, scheduler=None, scheduler_metric=None ,use_checkpoint=False, random_seed=0):

        self.use_checkpoint = use_checkpoint
        self.random_seed = random_seed

        if self.use_checkpoint:
            load_checkpoint(self, use_checkpoint)

        else:
            self.dataset = dataset
            if self.dataset == None:
                assert (dataloader_train, dataloader_valid) != (None, None), 'Error! Please make sure a dataset object or a train and valid dataloader objects are specified.'
                self.dataloader_train = dataloader_train
                self.dataloader_valid = dataloader_valid
            else:
                self.dataloader_train = self.dataset.dataloader_train
                self.dataloader_valid = self.dataset.dataloader_valid

            self.model = model
            self.criterion = criterion
            self.optimizer = optimizer
            if isinstance(scheduler, list):
                self.scheduler = scheduler
            else:
                self.scheduler = [scheduler]

            self.scheduler_metric = scheduler_metric
            self.device = select_device(device)

            if not name: self.name = current_time()+'_classifier_'
            else: self.name = name

            if str(type(model))[8:].startswith('sklearn'):

                self.type, self.feature_extractors = 'sklearn', {i:FeatureExtractor(model_arch=feature_extractor_arch, dataset=dataset, subset=i) for i in dataset.loaders.keys()}

                # self.type, self.feature_extractors = 'sklearn', {i:FeatureExtractor(model_arch=feature_extractor_arch, dataset=dataset, subset=i) for i in ['train', 'valid', 'test']}
                assert self.criterion == None, 'Error! Criterion cannot be used with sklearn classifier object type.'
                assert self.optimizer == None, 'Error! Optimizer cannot be used with sklearn classifier object type.'
            else:
                self.type = 'torch'

    def info(self):
        """
        Displays all information about the `ImageClassifier` object.
        """

        info=pd.DataFrame.from_dict(({key:str(value) for key, value in self.__dict__.items()}).items())
        info.columns=['Property', 'Value']
        for i in ['train', 'valid','test']:
            try:
                info.loc[len(info.index)] = [i+' dataset size', len(self.dataset.loaders[i].dataset)]
            except:
                pass
        return info

    def fit(self,**kwargs):
        """
        Trains the `ImageClassifier` object.

        !!! danger "Training a Model"
            All the following arguments, except `auto_save_ckpt` and `random_seed`, apply only when training a pytorch neural network model. Training sklearn classifier does not need arguments.

        Args:
            epochs (int): Number of training epochs (default: 20).
            valid (bool): True to perform validation after each train step. False to only train on training dataset without validation. (default: True)
            print_every (int): Number of epochs after which print results. (default: 1)
            target_valid_loss (float / string): Minimum value to automatically save trained model afterwards. If **'lowest'** is used, with every epoch , if the validation loss is less than minimum, then new best model is saved in checkpoint. Accepts maunally specified float minimum loss. (default: 'lowest')
            auto_save_ckpt (bool): Automatically save chekpoints. If True, a checkpoint file is saved. Please see below. (default: False)
            random_seed (int): Random seed. (default: 100)
            verbose (int): Verbose level during training. Options: 0, 1, 2. (default: 2)

        !!! warning "Using `auto_save_ckpt`"
            If `auto_save_ckpt` is True, whenever training target is achieved, a new checkpoint will be saved.

            The checkpoint file name = ```ImageClassifier.name+'epoch'+str(current_epoch)+'.checkpoint'```

            e.g. If the checkpoint is saved at epoch 10 for an `ImageClassifier` named clf, the checkpoint file will be named: clf_epoch_10.chekpoint

        !!! warning "Resuming training using a saved checkpoint file"
            When using a saved checkpoint to resume training, a new instance of the [`Model`](../model/#radtorch.model.Model)/Pytorch Model and [`ImageClassifier`](../classifier/#radtorch.classifier.ImageClassifier) should be instantiated.

            For example:

            ```python

            # Intial Training

            M =radtorch.model.Model(model_arch='vgg16', in_channels=1, out_classes=2)
            clf = radtorch.classifier.ImageClassifier(M, dataset)
            clf.fit(auto_save_ckpt=True, epochs=5, verbose=3) # Saves the best checkpoint automatically

            # Resume Training

            M =radtorch.model.Model(model_arch='vgg16', in_channels=1, out_classes=2)
            clf2 = radtorch.classifier.ImageClassifier(M, dataset, use_checkpoint='saved_ckpt.checkpoint')
            clf2.fit(auto_save_ckpt=False, epochs=5, verbose=3)
            ```



        !!! info "Checkpoint Files"
            **A checkpoint file is a dictionary of:**

            1. `timestamp`: Timestamp when saving the checkpoint.

            2. `type`: `ImageClassifier` type.

            3. `classifier`:  `ImageClassifier` object.

            4. `epochs`: Total epochs specified on initial training.

            5. `current_epoch`: Current epoch when checkpoint was saved.

            6.  `optimizer_state_dict`: Current state of Optimizer.

            7. `train_losses`: List of train losses recorded

            8. `valid_losses`: List of validation losses recorded

            9. `valid_loss_min`: Min Validation loss - See above.


        """

        if self.type == 'torch':
            kw = {k:v for k,v in kwargs.items() if k in fit_neural_network.__code__.co_varnames[:fit_neural_network.__code__.co_argcount]}
            fit_neural_network(self, **kw)

        elif self.type == 'sklearn':
            if hasattr(self, 'train_features'):
                message (' Using pre-extracted training features.')
            else:
                message (' Running Feature Extraction using model architecture '+str(self.feature_extractors['train'].model_arch))
                self.feature_extractors['train'].run()
            kw = {k:v for k,v in kwargs.items() if k in self._fit_sklearn.__code__.co_varnames[:self._fit_sklearn.__code__.co_argcount]}
            self._fit_sklearn(**kw)

    def _fit_sklearn(self, save_ckpt=False):
        message(" Starting model training on "+str(self.device))
        self.best_model = deepcopy(self.model)
        self.best_model.fit(self.feature_extractors['train'].features, self.feature_extractors['train'].labels)
        message(' Training completed successfully.')
        if save_ckpt:
            save_checkpoint(classifier=self, output_file=save_ckpt)
            message(' Trained model saved successfully.')

    def view_train_logs(self, data='all', figsize=(12,8)):
        assert self.type == 'torch', ('Train Logs not available with sklearn classifiers.')
        plt.figure(figsize=figsize)
        sns.set_style("darkgrid")
        if data == 'all': p = sns.lineplot(data = self.train_logs)
        else: p = sns.lineplot(data = self.train_logs[data].tolist())
        p.set_xlabel("epoch", fontsize = 10)
        p.set_ylabel("loss", fontsize = 10);

    def _are_models_equal(self, model_1, model_2):
        # https://discuss.pytorch.org/t/two-models-with-same-weights-different-results/8918/6
        models_differ = 0
        for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1

        if models_differ == 0:
            return True
        else:
            return False

    def sanity_check(self,):
        print ("Original model and training model weights are equal = {}".format(self._are_models_equal(self.original_model, self.model)))
        print ("Original model and best trained model weights are equal = {}".format(self._are_models_equal(self.original_model, self.best_model)))
        print ("Training model and best trained model weights are equal = {}".format(self._are_models_equal(self.model, self.best_model)))
