# Pipeline Module <small> radtorch.pipeline </small>


<p style='text-align: justify;'>
Pipelines are probably the most exciting feature of RADTorch Framework. With only few lines of code, the pipeline module allows you to run state-of-the-art machine learning algorithms and much more.
</p>


    from radtorch import pipeline



<p style='text-align: justify;'>
RADTorch follows principles of <b>object-oriented-programming</b> (OOP) in the sense that RADTorch pipelines are made of core building blocks and each of these blocks has specific functions/methods that can be accessed accordingly.
</p>


<p style='text-align: justify;'>

For example,
</p>

    pipeline.Image_Classification.data_processor.dataset_info()

<p style='text-align: justify;'>
can be used to access the dataset information for that particular Image Classification pipeline.
</p>

## Image_Classification
```
pipeline.Image_Classification(
              data_directory, is_dicom=False, table=None,
              image_path_column='IMAGE_PATH', image_label_column='IMAGE_LABEL',
              is_path=True, mode='RAW', wl=None, balance_class=False,
              balance_class_method='upsample', interaction_terms=False,
              normalize=((0,0,0), (1,1,1)), batch_size=16, num_workers=0,
              sampling=1.0, test_percent=0.2, valid_percent=0.2, custom_resize=False,
              model_arch='alexnet', pre_trained=True, unfreeze=False,
              type='nn_classifier', cv=True, stratified=True, num_splits=5, parameters={},
              learning_rate=0.0001, epochs=10, optimizer='Adam',
              loss_function='CrossEntropyLoss',lr_scheduler=None,
              custom_nn_classifier=None, loss_function_parameters={},
              optimizer_parameters={}, transformations='default',
              extra_transformations=None, device='auto',)
```

!!! quote ""


      **Description**

      Complete end-to-end image classification pipeline.

      **Parameters**

      - data_directory (string, required): path to target data directory/folder.

      - is_dicom (bollean, optional): True if images are DICOM. default=False.

      - table (string or pandas dataframe, optional): path to label table csv or name of pandas data table. default=None.

      - image_path_column (string, optional): name of column that has image path/image file name. default='IMAGE_PATH'.

      - image_label_column (string, optional): name of column that has image label. default='IMAGE_LABEL'.

      - is_path (boolean, optional): True if file_path column in table is file path. If False, this assumes that the column contains file names only and will append the data_directory to all files. default=True.

      - mode (string, optional): mode of handling pixel values from DICOM to numpy array. Option={'RAW': raw pixel values, 'HU': converts pixel values to HU using slope and intercept, 'WIN':Applies a certain window/level to HU converted DICOM image, 'MWIN': converts DICOM image to 3 channel HU numpy array with each channel adjusted to certain window/level. default='RAW'.

      - wl (tuple or list of tuples, optional): value of Window/Levelto be used. If mode is set to 'WIN' then wl takes the format (level, window). If mode is set to 'MWIN' then wl takes the format [(level1, window1), (level2, window2), (level3, window3)]. default=None.

      - balance_class (bollean, optional): True to perform oversampling in the train dataset to solve class imbalance. default=False.

      - balance_class_method (string, optional): methodology used to balance classes. Options={'upsample', 'downsample'}. default='upsample'.

      - interaction_terms (boolean, optional): create interaction terms between different features and add them as new features to feature table. default=False.

      - normalize (bolean/False or Tuple, optional): Normalizes all datasets by a specified mean and standard deviation. Since most of the used CNN architectures assumes 3 channel input, this follows the following format ((mean, mean, mean), (std, std, std)). default=((0,0,0), (1,1,1)).

      - batch_size (integer, optional): Batch size for dataloader. defult=16.

      - num_workers (integer, optional): Number of CPU workers for dataloader. default=0.

      - sampling (float, optional): fraction of the whole dataset to be used. default=1.0.

      - test_percent (float, optional): percentage of data for testing.default=0.2.

      - valid_percent (float, optional): percentage of data for validation (ONLY with NN_Classifier) .default=0.2.

      - custom_resize (integer, optional): By default, the data processor resizes the image in dataset into the size expected bu the different CNN architectures. To override this and use a custom resize, set this to desired value. default=False.

      - transformations (list, optional): list of pytorch transformations to be applied to all datasets. By default, the images are resized, channels added up to 3 and greyscaled. default='default'.

      - extra_transformations (list, optional): list of pytorch transformations to be extra added to train dataset specifically. default=None.

      - model_arch (string, required): CNN model architecture that this data will be used for. Used to resize images as detailed above. default='alexnet' .

      - pre_trained (boolean, optional): Initialize with ImageNet pretrained weights or not. default=True.

      - unfreeze (boolean, required): Unfreeze all layers of network for future retraining. default=False.

      - type (string, required): type of classifier. For complete list refer to settings. default='logistic_regression'.

      ***Classifier specific parameters***

      - cv (boolean, required): True for cross validation. default=True.

      - stratified (boolean, required): True for stratified cross validation. default=True.

      - num_splits (integer, required): Number of K-fold cross validation splits. default=5.

      - parameters (dictionary, optional): optional parameters passed to the classifier. Please refer to sci-kit learn documentaion.

      ***NN_Classifier specific parameters***

      - learning_rate (float, required): Learning rate. default=0.0001.

      - epochs (integer, required): training epochs. default=10.

      - optimizer (string, required): neural network optimizer type. Please see radtorch.settings for list of approved optimizers. default='Adam'.

      - optimizer_parameters (dictionary, optional): optional extra parameters for optimizer as per pytorch documentation.

      - loss_function (string, required): neural network loss function. Please see radtorch.settings for list of approved loss functions. default='CrossEntropyLoss'.

      - loss_function_parameters (dictionary, optional): optional extra parameters for loss function as per pytorch documentation.

      - lr_scheduler (string, optional): learning rate scheduler - upcoming soon.

      - custom_nn_classifier (pytorch model, optional): Option to use a custom made neural network classifier that will be added after feature extracted layers. default=None.

      - device (string, optional): device to be used for training. Options{'auto': automatic detection of device type, 'cpu': cpu, 'cuda': gpu}. default='auto'.

      **Methods**

      - In addition to [core component methods](https://www.radtorch.com/documentation/core/), pipeline accessible methods include:
      ```
      .info()
      ```
        Displays information of the image classification pipeline.
      ```
      .run()
      ```
        Starts the image classification pipeline training.
      ```
      .metrics(figure_size=(700, 350))
      ```
        Displays the training metrics of the image classification pipeline.
      ```
      .export(output_path):
      ```
        Exports the pipeline to output path.




## GAN

```
core.GAN(
        data_directory, table=None, is_dicom=False, is_path=True,
        image_path_column='IMAGE_PATH', image_label_column='IMAGE_LABEL',
        mode='RAW', wl=None, batch_size=16, normalize=((0,0,0),(1,1,1)),
        num_workers=0, label_smooth=True, sampling=1.0, transformations='default',

        discriminator='dcgan', generator='dcgan',
        generator_noise_size=100, generator_noise_type='normal',
        discriminator_num_features=64, generator_num_features=64,
        image_size=128, image_channels=1,

        discrinimator_optimizer='Adam', generator_optimizer='Adam',
        discrinimator_optimizer_param={'betas':(0.5,0.999)},
        generator_optimizer_param={'betas':(0.5,0.999)},
        generator_learning_rate=0.0001, discriminator_learning_rate=0.0001,        

        epochs=10, device='auto')


```

!!! quote ""


      **Description**

      Generative Advarsarial Networks Pipeline.


      **Parameters**

      - data_directory (string, required): path to target data directory/folder.

      - is_dicom (bollean, optional): True if images are DICOM. default=False.

      - table (string or pandas dataframe, optional): path to label table csv or name of pandas data table. default=None.

      - image_path_column (string, optional): name of column that has image path/image file name. default='IMAGE_PATH'.

      - image_label_column (string, optional): name of column that has image label. default='IMAGE_LABEL'.

      - is_path (boolean, optional): True if file_path column in table is file path. If False, this assumes that the column contains file names only and will append the data_directory to all files. default=True.

      - mode (string, optional): mode of handling pixel values from DICOM to numpy array. Option={'RAW': raw pixel values, 'HU': converts pixel values to HU using slope and intercept, 'WIN':Applies a certain window/level to HU converted DICOM image, 'MWIN': converts DICOM image to 3 channel HU numpy array with each channel adjusted to certain window/level. default='RAW'.

      - wl (tuple or list of tuples, optional): value of Window/Levelto be used. If mode is set to 'WIN' then wl takes the format (level, window). If mode is set to 'MWIN' then wl takes the format [(level1, window1), (level2, window2), (level3, window3)]. default=None.

      - batch_size (integer, optional): Batch size for dataloader. defult=16.

      - num_workers (integer, optional): Number of CPU workers for dataloader. default=0.

      - sampling (float, optional): fraction of the whole dataset to be used. default=1.0.

      - transformations (list, optional): list of pytorch transformations to be applied to all datasets. By default, the images are resized, channels added up to 3 and greyscaled. default='default'.

      - normalize (bolean/False or Tuple, optional): Normalizes all datasets by a specified mean and standard deviation. Since most of the used CNN architectures assumes 3 channel input, this follows the following format ((mean, mean, mean), (std, std, std)). default=((0,0,0),(1,1,1)).

      - label_smooth (boolean, optioanl): by default, labels for real images as assigned to 1. If label smoothing is set to True, lables of real images will be assigned to 0.9. default=True. (Source: https://github.com/soumith/ganhacks#6-use-soft-and-noisy-labels)

      - epochs (integer, required): training epochs. default=10.

      - generator (string, required): type of generator network. Options = {'dcgan', 'vanilla'}. default='dcgan'

      - discriminator (string, required): type of discriminator network. Options = {'dcgan', 'vanilla'}. default='dcgan'

      - image_channels (integer, required): number of output channels for discriminator input and generator output. default=1

      - generator_noise_type (string, optional): shape of noise to sample from. Options={'normal', 'gaussian'}. default='normal'. (https://github.com/soumith/ganhacks#3-use-a-spherical-z)

      - generator_noise_size (integer, required): size of the noise sample to be generated. default=100

      - generator_num_features (integer, required): number of features/convolutions for generator network. default=64

      - image_size (integer, required): iamge size for discriminator input and generator output.default=128

      - discriminator_num_features (integer, required): number of features/convolutions for discriminator network.default=64

      - generator_optimizer (string, required): generator network optimizer type. Please see radtorch.settings for list of approved optimizers. default='Adam'.

      - generator_optimizer_param (dictionary, optional): optional extra parameters for optimizer as per pytorch documentation. default={'betas':(0.5,0.999)} for Adam optimizer.

      - discrinimator_optimizer (string, required): discrinimator network optimizer type. Please see radtorch.settings for list of approved optimizers. default='Adam'.

      - discrinimator_optimizer_param (dictionary, optional): optional extra parameters for optimizer as per pytorch documentation. default={'betas':(0.5,0.999)} for Adam optimizer.

      - generator_learning_rate (float, required): generator network learning rate. default=0.0001.

      - discriminator_learning_rate (float, required): discrinimator network learning rate. default=0.0001.

      - device (string, optional): device to be used for training. Options{'auto': automatic detection of device type, 'cpu': cpu, 'cuda': gpu}. default='auto'.



      **Methods**
      ```
      .run(self, verbose='batch', show_images=True, figure_size=(10,10))
      ```

      - Runs the GAN training.

      - Parameters:

          - verbose (string, required): amount of data output. Options {'batch': display info after each batch, 'epoch': display info after each epoch}.default='batch'

          - show_images (boolean, optional): True to show sample of generatot generated images after each epoch.

          - figure_size (tuple, optional): Tuple of width and length of figure plotted. default=(10,10)


      ```
      .sample(figure_size=(10,10), show_labels=True)
      ```

      - Displays a sample of real data.

      - Parameters:

          - figure_size (tuple, optional): Tuple of width and length of figure plotted. default=(10,10).

          - show_labels (boolean, optional): show labels on top of images. default=True.

      ```
      .info()
      ```

      - Displays different parameters of the generative adversarial network.

      ```
      .metrics(figure_size=(700,350))
      ```

      - Displays training metrics for the GAN.

      - Explanation of metrics:

          - *D_loss*: Total loss of discriminator network on both real and fake images.

          - *G_loss*: Loss of discriminator network on detecting fake images as real.

          - *d_loss_real*: Loss of discriminator network on detecting real images as real.

          - *d_loss_fake*: Loss of discriminator network on detecting fake images as fake.

      - Parameters:

          - figure_size (tuple, optional): Tuple of width and length of figure plotted. default=(700,350).



<small> Documentation Update: 5/14/2020 </small>
