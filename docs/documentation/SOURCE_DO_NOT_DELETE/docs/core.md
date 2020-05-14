
# Core Module <small> radtorch.core </small>


```
from radtorch import core
```
The core module has all the core functionalities of RADTorch framework. These include:

1. RADTorch_Dataset

2. Data_Processor

3. Feature_Extractor

4. Classifier

5. NN_Classifier

6. DCGAN_Discriminator

7. DCGAN_Generator

8. GAN_Discriminator

9. GAN_Generator

10. Feature_Selector (Coming Soon)



## Image classification

### RADTorch_Dataset
    core.RADTorch_Dataset(data_directory,transformations,
                          table=None,is_dicom=False,
                          mode='RAW', wl=None,
                          image_path_column='IMAGE_PATH',
                          image_label_column='IMAGE_LABEL',
                          is_path=True, sampling=1.0)

!!! quote ""

    **Description**

    Core class for dataset. This is an extension of Pytorch dataset class with modifications.

    **Parameters**

    - data_directory (string, required): path to target data directory/folder.

    - is_dicom (bollean, optional): True if images are DICOM. default=False.

    - table (string or pandas dataframe, optional): path to label table csv or name of pandas data table. default=None.

    - image_path_column (string, optional): name of column that has image path/image file name. default='IMAGE_PATH'.

    - image_label_column (string, optional): name of column that has image label. default='IMAGE_LABEL'.

    - is_path (boolean, optional): True if file_path column in table is file path. If False, this assumes that the column contains file names only and will append the data_directory to all files. default=True.

    - mode (string, optional): mode of handling pixel values from DICOM to numpy array. Option={'RAW': raw pixel values, 'HU': converts pixel values to HU using slope and intercept, 'WIN':Applies a certain window/level to HU converted DICOM image, 'MWIN': converts DICOM image to 3 channel HU numpy array with each channel adjusted to certain window/level. default='RAW'.

    - wl (tuple or list of tuples, optional): value of Window/Levelto be used. If mode is set to 'WIN' then wl takes the format (level, window). If mode is set to 'MWIN' then wl takes the format [(level1, window1), (level2, window2), (level3, window3)]. default=None.

    - sampling (float, optional): fraction of the whole dataset to be used. default=1.0.

    - transformations (list, optional): list of pytorch transformations to be applied to all datasets. By default, the images are resized, channels added up to 3 and greyscaled. default='default'.


    **Returns**

    RADTorch dataset object.


    **Methods**

    *info()*

    - Returns information of the dataset.

    *.classes()*

    - Returns list of classes in dataset.

    *.class_to_idx()*

    - Returns mapping of classes to class id (dictionary).

    *.parameters()*

    - Returns all the parameter names of the dataset.

    *.balance(method='upsample')*

    - Returns a balanced dataset. methods={'upsample', 'downsample'}

    *.mean_std()*

    - calculates mean and standard deviation of dataset. Returns tuple of (mean, std)

    *.normalize()*

    - Returns a normalized dataset with either mean/std of the dataset or a user specified mean/std in the form of ((mean, mean, mean), (std, std, std)).



### Data_Processor

    core.Data_Processor(data_directory,is_dicom=False,table=None,
                        image_path_column='IMAGE_PATH',
                        image_label_column='IMAGE_LABEL',
                        is_path=True, mode='RAW', wl=None,
                        balance_class=False, balance_class_method='upsample',
                        normalize=((0,0,0), (1,1,1)), batch_size=16,
                        num_workers=0, sampling=1.0, custom_resize=False,
                        model_arch='alexnet', type='nn_classifier',
                        transformations='default',
                        extra_transformations=None,
                        test_percent=0.2, valid_percent=0.2, device='auto')


!!! quote ""

    **Description**

    Class Data Processor. The core class for data preparation before feature extraction and classification. This class performs dataset creation, data splitting, sampling, balancing, normalization and transformations.


    **Parameters**

    - data_directory (string, required): path to target data directory/folder.

    - is_dicom (bollean, optional): True if images are DICOM. default=False.

    - table (string or pandas dataframe, optional): path to label table csv or name of pandas data table. default=None. **None** means the Data_Processor will create the datasets and labels from folder structure as shown [here](https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.ImageFolder).

    - image_path_column (string, optional): name of column that has image path/image file name. default='IMAGE_PATH'.

    - image_label_column (string, optional): name of column that has image label. default='IMAGE_LABEL'.

    - is_path (boolean, optional): True if file_path column in table is file path. If False, this assumes that the column contains file names only and will append the data_directory to all files. default=False.

    - mode (string, optional): mode of handling pixel values from DICOM to numpy array. Option={'RAW': raw pixel values, 'HU': converts pixel values to HU using slope and intercept, 'WIN':Applies a certain window/level to HU converted DICOM image, 'MWIN': converts DICOM image to 3 channel HU numpy array with each channel adjusted to certain window/level. default='RAW'.

    - wl (tuple or list of tuples, optional): value of Window/Levelto be used. If mode is set to 'WIN' then wl takes the format (level, window). If mode is set to 'MWIN' then wl takes the format [(level1, window1), (level2, window2), (level3, window3)]. default=None.

    - balance_class (bollean, optional): True to perform oversampling in the train dataset to solve class imbalance. default=False.

    - balance_class_method (string, optional): methodology used to balance classes. Options={'upsample', 'downsample'}. default='upsample'.

    - normalize (bollean, optional): Normalizes all datasets by a specified mean and standard deviation. Since most of the used CNN architectures assumes 3 channel input, this follows the following format ((mean, mean, mean), (std, std, std)). default=False.

    - batch_size (integer, optional): Batch size for dataloader. defult=16.

    - num_workers (integer, optional): Number of CPU workers for dataloader. default=0.

    - sampling (float, optional): fraction of the whole dataset to be used. default=1.0.

    - test_percent (float, optional): percentage of data for testing.default=0.2.

    - valid_percent (float, optional): percentage of data for validation (ONLY with NN_Classifier) .default=0.2.

    - custom_resize (integer, optional): By default, the data processor resizes the image in dataset into the size expected bu the different CNN architectures. To override this and use a custom resize, set this to desired value. default=False.

    - model_arch (string, required): CNN model architecture that this data will be used for. Used to resize images as detailed above. default='alexnet' .

    - type (string, required): type of classifier that will be used. please refer to classifier object type. default='nn_classifier'.

    - device (string, optional): device to be used for training. Options{'auto': automatic detection of device type, 'cpu': cpu, 'cuda': gpu}. default='auto'.

    - transformations (list, optional): list of pytorch transformations to be applied to all datasets. By default, the images are resized, channels added up to 3 and greyscaled. default='default'.

    - extra_transformations (list, optional): list of pytorch transformations to be extra added to train dataset specifically. default=None.


    **Methods**

    *.classes()*

    - Returns dictionary of classes/class_idx in data.

    *.info()*

    - Returns full information of the data processor object.

    *.dataset_info(plot=True, figure_size=(500,300))*

    - Displays information of the data and class breakdown.

    - Parameters:

        - plot (boolean, optional): True to display data as graph. False to display in table format. default=True

        - figure_size (tuple, optional): Tuple of width and length of figure plotted. default=(500,300)


    *.sample(figure_size=(10,10), show_labels=True, show_file_name=False)*

    - Displays a sample from the training dataset. Number of images displayed is the same as batch size.

    - Parameters:

        - figure_size (tuple, optional): Tuple of width and length of figure plotted. default=(10,10)

        - show_label (boolean, optional): show labels above images. default=True

        - show_file_names (boolean, optional): show file path above image. default=False


    *.check_leak(show_file=False)*

    - Checks possible overlap between train and test dataset files.

    - Parameters:

        - show_file (boolean, optional): display table of leaked/common files between train and test. default=False.


    *.export(output_path)*

    - Exports the Dtaprocessor object for future use.

    - Parameters:

        - output_path (string, required): output file path.  




### Feature_Extractor

    core.Feature_Extractor(model_arch, dataloader,pre_trained=True, unfreeze=False,
                           device='auto',)


!!! quote ""

    Creates a feature extractor neural network using one of the famous CNN architectures and the data provided as dataloader from Data_Processor.

    **Parameters**

    - model_arch (string, required): CNN architecture to be utilized. To see list of supported architectures see settings.

    - pre_trained (boolean, optional): Initialize with ImageNet pretrained weights or not. default=True.

    - unfreeze (boolean, required): Unfreeze all layers of network for future retraining. default=False.

    - dataloader (pytorch dataloader object, required): the dataloader that will be used to supply data for feature extraction.

    - device (string, optional): device to be used for training. Options{'auto': automatic detection of device type, 'cpu': cpu, 'cuda': gpu}. default='auto'.

    <div align='center'>

    | Model Architecture | Default Input Image Size | Output Features |
    |--------------------|:------------------------:|:---------------:|
    | vgg11              |         224 x 224        |       4096      |
    | vgg13              |         224 x 224        |       4096      |
    | vgg16              |         224 x 224        |       4096      |
    | vgg19              |         224 x 224        |       4096      |
    | vgg11_bn           |         224 x 224        |       4096      |
    | vgg13_bn           |         224 x 224        |       4096      |
    | vgg16_bn           |         224 x 224        |       4096      |
    | vgg19_bn           |         224 x 224        |       4096      |
    | resnet18           |         224 x 224        |       512       |
    | resnet34           |         224 x 224        |       512       |
    | resnet50           |         224 x 224        |       2048      |
    | resnet101          |         224 x 224        |       2048      |
    | resnet152          |         224 x 224        |       2048      |
    | wide_resnet50_2    |         224 x 224        |       2048      |
    | wide_resnet101_2   |         224 x 224        |       2048      |
    | alexnet            |         256 x 256        |       4096      |

    </div>

    **Returns**

      Pandas dataframe with extracted features.

    **Methods**  

    *.num_features()*

    - Returns the number of features to be extracted.

    *.run()*

    - Runs the feature extraction process. Returns tuple of feature_table (dataframe which contains all features, labels and image file path), features (dataframe which contains features only), feature_names(list of feature names)


    *.export_features(csv_path)*

    - Exports extracted features into csv file.

    - Parameters:

          - csv_path (string, required): path to csv output.

    *.plot_extracted_features(num_features=100, num_images=100,image_path_column='IMAGE_PATH', image_label_column='IMAGE_LABEL')*

    - Plots Extracted Features in Heatmap

    - Parameters:

          - num_features (integer, optional): number of features to display. default=100

          - num_images (integer, optional): number of images to display features for. default=100

          - image_path_column (string, required): name of column that has image names/path. default='IMAGE_PATH'

          - image_label_column (string, required): name of column that has image labels. default='IMAGE_LABEL'


    *.export(output_path)*

    - Exports the Feature Extractor object for future use.

    - Parameters:

        -  output_path (string, required): output file path.



### Classifier

    core.Classifier(extracted_feature_dictionary, feature_table=None,
                    image_label_column=None, image_path_column=None,
                    test_percent=None, type='logistic_regression',
                    interaction_terms=False, parameters={},
                    cv=True, stratified=True, num_splits=5)

!!! quote ""

    **Description**

    Image Classification Class. Performs Binary/Multiclass classification using features extracted via Feature Extractor or Supplied by user.


    **Parameters**


    - extracted_feature_dictionary (dictionary, required): Dictionary of features/labels datasets to be used for classification. This follows the following format :
    {
        'train':
                {'features':dataframe, 'feature_names':list, 'labels': list}},
        'test':
                {'features':dataframe, 'feature_names':list, 'labels': list}},
    }

    - feature_table (string, optional): path to csv table with user selected image paths, labels and features. default=None.

    - image_label_column (string, required if using feature_table): name of the column with images labels.default=None.

    - image_path_column (string, requried if using feature_table): name of column with images paths.default=None.

    - test_percent (float, required if using feature_table): percentage of data for testing.default=None.

    - type (string, required): type of classifier. For complete list refer to settings. default='logistic_regression'.

    - interaction_terms (boolean, optional): create interaction terms between different features and add them as new features to feature table. default=False.

    - cv (boolean, required): True for cross validation. default=True.

    - stratified (boolean, required): True for stratified cross validation. default=True.

    - num_splits (integer, required): Number of K-fold cross validation splits. default=5.

    - parameters (dictionary, optional): optional parameters passed to the classifier. Please refer to sci-kit learn documentation.


    **Methods**


    *.info()*

    - Returns table of different classifier parameters/properties.

    *.run()*

    - Runs Image Classifier.

    *.average_cv_accuracy()*

    - Returns average cross validation accuracy.

    *.test_accuracy()*

    - Returns accuracy of trained classifier on test dataset.

    *.confusion_matrix(title='Confusion Matrix',cmap=None,normalize=False,figure_size=(8,6))*

    - Displays confusion matrix using trained classifier and test dataset.

    - Parameters:

        - title (string, optional): name to be displayed over confusion matrix.

        - cmap (string, optional): colormap of the displayed confusion matrix. This follows matplot color palletes. default=None.

        - normalize (boolean, optional): normalize values. default=False.

        - figure_size (tuple, optional): size of the figure as width, height. default=(8,6)


    *.roc()*

    - Display ROC and AUC of trained classifier and test dataset.

    *.predict(input_image_path, all_predictions=False)*

    - Returns label prediction of a target image using a trained classifier. This works as part of pipeline only for now.

    - Parameters:

        - input_image_path (string, required): path of target image.

        - all_predictions (boolean, optional): return a table of all predictions for all possible labels.


    *.export()*

    - Exports the Classifier object for future use. output_path (string, required): output file path.

    *.export_trained_classifier()*

    - Exports only the trained classifier for future use. output_path (string, required): output file path.







### NN_Classifier

    core.NN_Classifier(feature_extractor, data_processor, unfreeze=False,
                      learning_rate=0.0001, epochs=10, optimizer='Adam',
                      loss_function='CrossEntropyLoss', lr_scheduler=None,
                      batch_size=16, device='auto', custom_nn_classifier=None,
                      loss_function_parameters={}, optimizer_parameters={},)


!!! quote ""

    **Description**

    Neural Network Classifier. This serves as extension of pytorch neural network modules e.g. VGG16, for fine tuning or transfer learning.


    **Parameters**

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

    **Methods**

    *.info()*

    - Returns table of different classifier parameters/properties.

    *.run()*

    - Runs Image Classifier.

    *.confusion_matrix(target_dataset=None, figure_size=(8,6), cmap=None)*

    - Displays confusion matrix for trained nn_classifier on test dataset.

    - Parameters:

        - target_dataset (pytorch dataset, optional): this option can be used to test the trained model on an external test dataset. If set to None, the confusion matrix is generated using the test dataset initially specified in the data_processor. default=None.

        - figure_size (tuple, optional): size of the figure as width, height. default=(8,6)

    *.roc()*

    - Display ROC and AUC of trained model and test dataset.

    *.metrics(figure_size=(700,400))*

    - Displays graphical representation of train/validation loss /accuracy.

    *.predict(input_image_path, all_predictions=True)*

    - Displays classs prediction for a target image using a trained classifier.

    - Parameters:

        - input_image_path (string, required): path to target image.

        - all_predictions (boolean, optional): True to display prediction percentage accuracies for all prediction classes. default=True.


    *.misclassified(num_of_images=4, figure_size=(5,5), table=False)*

    - Displays sample of images misclassified by the classifier from test dataset.

    - Parameters:

        - num_of_images (integer, optional): number of images to be displayed. default=4.

        - figure_size (tuple, optional): size of the figure as width, height. default=(5,5).

        - table (boolean, optional): True to display a table of all misclassified images including image path, true label and predicted label.



## Generative Adversarial Networks

### DCGAN_Discriminator

    core.DCGAN_Discriminator(num_input_channels, kernel_size, num_discriminator_features,
                             input_image_size, device='auto')


!!! quote ""

     **Description**

     Core Deep Convolutional GAN Discriminator Network.


     **Parameters**

     - kernel_size (integer, required): size of kernel/filter to be used for convolution.

     - num_discriminator_features (integer, required): number of features/convolutions for discriminator network.

     - num_input_channels (integer, required): number of channels for input image.

     - input_image_size (integer, required): size of input image.

     - device (string, optional): device to be used for training. Options{'auto': automatic detection of device type, 'cpu': cpu, 'cuda': gpu}. default='auto'.



### DCGAN_Generator

    core.DCGAN_Generator(noise_size, num_generator_features, num_output_channels,
                        target_image_size, device='auto')


!!! quote ""

      **Description**

      Core Deep Convolutional GAN Generator Network.


      **Parameters**

      - noise_size (integer, required): size of the noise sample to be generated.

      - num_generator_features (integer, required): number of features/convolutions for generator network.

      - num_output_channels (integer, required): number of channels for output image.

      - target_image_size (integer, required): size of output image.

      - device (string, optional): device to be used for training. Options{'auto': automatic detection of device type, 'cpu': cpu, 'cuda': gpu}. default='auto'.


### GAN_Discriminator

      core.GAN_Discriminator(input_image_size, intput_num_channels, device='auto')


!!! quote ""      

      **Description**

      Core Vanilla GAN Discriminator Network.


      **Parameters**

      - num_input_channels (integer, required): number of channels for input image.

      - input_image_size (integer, required): size of input image.

      - device (string, optional): device to be used for training. Options{'auto': automatic detection of device type, 'cpu': cpu, 'cuda': gpu}. default='auto'.


### GAN_Generator

      core.GAN_Generator(noise_size, target_image_size, output_num_channels, device='auto')


!!! quote ""  

      **Description**

      Core Vanilla Convolutional GAN Generator Network.


      **Parameters**

      - noise_size (integer, required): size of the noise sample to be generated.

      - num_output_channels (integer, required): number of channels for output image.

      - target_image_size (integer, required): size of output image.

      - device (string, optional): device to be used for training. Options{'auto': automatic detection of device type, 'cpu': cpu, 'cuda': gpu}. default='auto'.


<small> Documentation Update: 5/14/2020 </small>
