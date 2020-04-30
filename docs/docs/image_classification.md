
```
pipeline.Image_Classification(
    data_directory, name=None, table=None, is_dicom=True, custom_resize=False,
    test_percent=0.2, valid_percent=0.2, balance_class=False, batch_size=16,
    normalize=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), num_workers=1,
    model_arch='alexnet', pre_trained=True, unfreeze=False,
    type='ridge', cv=True, stratified=True, num_splits=5, parameters={})

```  



The Image Classification pipeline simplifies the process of binary and multi-class image classification into a single line of code.
Under the hood, the following happens:

1. The pipeline creates a master dataset from the provided data directory and source of labels/classes either from [folder structure](https://pytorch.org/docs/stable/torchvision/datasets.html#datasetfolder) or pandas/csv table.

2. Master dataset is subdivided into train, valid and test subsets using the percentages defined by user.

3. The following transformations are applied on the dataset images:
    1. Resize to the default image size allowed by the model architecture.
    2. Window/Level adjustment according to values specified by user.
    3. Single channel grayscale DICOM images are converted into 3 channel grayscale images to fit into the model.

3. Selected Model architecture, optimizer, and loss function are downloaded/created.

4. Model is trained.

5. Training metrics are saved as training progresses and can be displayed after training is done.

6. Confusion Matrix and ROC (for binary classification) can be displayed as well (by default, the test subset is used to calculate the confusion matrix and the ROC)

7. Misclassifed samples can be displayed.

8. Trained model can be exported to outside file for future use.


<!-- ####Parameters -->

??? quote  "Parameters"

    **data_directory:**

    - _(str)_ target data directory. **(Required)**

    **name:**

    - _(str)_ preferred name to be given to classifier.

    **is_dicom:**

    - _(boolean)_ True for DICOM images. (default=True)

    **label_from_table:**

    - _(boolean)_ True if labels are to extracted from table, False if labels are to be extracted from subfolders names. (default=False)

    **is_csv:**

    - _(boolean)_ True for csv, False for pandas dataframe.

    **table_source:**

    - _(str or pandas dataframe object)_ source for labelling data.This is path to csv file or name of pandas dataframe if pandas to be used. (default=None).

    **predefined_datasets**

    - _(dict)_ dictionary of predefined pandas dataframes for training. This follows the following scheme: {'train': dataframe, 'valid': dataframe, 'test':dataframe }

    **path_col:**

    - _(str)_  name of the column with the image path. (default='IMAGE_PATH')

    **label_col:**

    - _(str)_  name of the label/class column. (default='IMAGE_LABEL')

    **mode:**

    - _(str)_  output mode for DICOM images only where RAW= Raw pixels, HU= Image converted to Hounsefield Units, WIN= 'window' image windowed to certain W and L, MWIN = 'multi-window' converts image to 3 windowed images of different W and L (specified in wl argument) stacked together. (default='RAW')

    **wl:**

    - _(list)_ list of lists of combinations of window level and widths to be used with WIN and MWIN.In the form of : [[Level,Width], [Level,Width],...].  
    - Only 3 combinations are allowed for MWIN (for now). (default=None)

    **balance_class:**

    - _(boolen)_ balance classes in train/valid/test subsets. Under the hood, oversampling is done for the classes with fewer number of instances. (default=False)

    **normalize:**

    - _(str)_ Normalization algorithm applied to data. Options: 'default' normalizes data with mean of 0.5 and standard deviation of 0.5, 'auto' normalizes the data using mean and standard deviation calculated from the datasets, 'False' applies no normalization. (default='default')

    **transformations:**

    - _(pytorch transforms list)_ pytroch transforms to be performed on the dataset. (default=Convert to tensor)

    **custom_resize:**
    - _(int)_ by default, a radtorch pipeline will resize the input images into the default training model input image size as demonstrated in the table shown below. This default size can be changed here if needed.

    **batch_size:**

    - _(int)_ batch size of the dataset (default=16)

    **test_percent:**

    - _(float)_ percentage of dataset to use for testing. Float value between 0 and 1.0. (default=0.2)

    **valid_percent:**

    - _(float)_ percentage of dataset to use for validation. Float value between 0 and 1.0. (default=0.2)

    **model_arch:**

    - _(str)_ PyTorch neural network architecture (default='vgg16')

    **pre_trained:**

    - _(boolean)_ Load the pretrained weights of the neural network. (default=True)

    **unfreeze_weights:**

    - _(boolean)_ if True, all model weights will be retrained. This note that if no pre_trained weights are applied, this option will be set to True automatically. (default=False)

    **train_epochs:**

    - _(int)_ Number of training epochs. (default=20)

    **learning_rate:**

    - _(str)_ training learning rate. (default = 0.0001)

    **loss_function:**

    - _(str)_ training loss function. (default='CrossEntropyLoss')

    **optimizer:**

    - _(str)_ Optimizer to be used during training. (default='Adam')

    **device:**

    - _(str)_ device to be used for training. This can be adjusted to 'cpu' or 'cuda'. If nothing is selected, the pipeline automatically detects if CUDA is available and trains on it.


<!-- ####Methods -->

??? quote "Methods"


    **.info()**

    - Display table with properties of the Image Classification Pipeline.

    **.dataset_info(plot=True, plot_size=(500,300))**

    - Display Dataset Information.

    - Arguments:
        - plot: _(boolean)_ displays dataset information in graphical representation. (default=True)
        - plot_size: _(tuple)_ figures size.

    **.sample(fig_size=(10,10), show_labels=True, show_file_name=False)**

    - Display sample of the training dataset.

    - Arguments:
        - fig_size: _(tuple)_ figure size. (default=(10,10))
        - show_labels: _(boolean)_ show the image label idx. (default=True)
        - show_file_name: _(boolean)_ show the file name as label. (default=False)

    **.run(verbose=True)**

    - Start the image classification pipeline training.

    - Arguments:
        - verbose: _(boolean)_ Show display progress after each epoch. (default=True)

    **.metrics(fig_size=(500,300))**

    - Display the training metrics.

    **.export_model(output_path)**

    - Export the trained model into a target file.

    - Arguments:
        - output_path: _(str)_ path to output file. For example 'foler/folder/model.pth'

    **.export(output_path)**

    - Exports the whole image classification pipeline for future use

    - Arguments:
        - target_path: _(str)_ target location for export.


    **.set_trained_model(model_path, mode)**

    - Loads a previously trained model into pipeline

    - Arguments:
        - model_path: _(str)_ path to target model
        - mode: _(str)_ either 'train' or 'infer'.'train' will load the model to be trained. 'infer' will load the model for inference.


    **.inference(test_img_path, transformations='default',  all_predictions=False)**

    - Performs inference using the trained model on a target image.

    - Arguments:
        - test_img_path: _(str)_ path to target image.
        - transformations: _(pytorch transforms list)_ list of transforms to be performed on the target image. (default='default' which is the same transforms using for training the pipeline)
        - all_predictions: _(boolean)_  if True , shows table of all prediction classes and accuracy percentages. (default=False)


    **.roc(target_data_set='default', figure_size=(600,400))**

    - Display ROC and AUC.

    - Arguments:
        - target_data_set: _(pytorch dataset object)_ dataset used for predictions to create the ROC. By default, the image classification pipeline uses the test dataset created to calculate the ROC. If no test dataset was created in the pipeline (e.g. test_percent=0), then an external test dataset is required. (default=default')
        - figure_size: _(tuple)_ figure size. (default=(600,400))


    **.confusion_matrix(target_data_set='default', target_classes='default', figure_size=(7,7), cmap=None)**

    - Display Confusion Matrix

    - Arguments:
        - target_data_set: _(pytorch dataset object)_ dataset used for predictions to create the confusion matrix. By default, the image classification - pipeline uses the test dataset created to calculate the matrix.
        - target_classes: _(list)_ list of classes. By default, the image classification pipeline uses the training classes.
        - figure_size: _(tuple)_ figure size. (default=(7,7))
        - cmap: _(str)_ user specific matplotlib cmap.


    **.misclassfied(target_data_set='default', num_of_images=16, figure_size=(10,10), show_table=False)**

    - Displays sample of misclassfied images from confusion matrix or ROC.

    - Arguments:
      - target_data_set: _(pytorch dataset object)_ dataset used for predictions. By default, the image classification pipeline uses the test dataset. If no test dataset was created in the pipeline (e.g. test_percent=0), then an external test dataset is required. (default=default')
      - num_of_images: _(int)_ number of images to be displayed.
      - figure_size: _(tuple)_ figure size (default=(10,10))
      - show_table: _(boolean)_ display table of misclassied images. (default=False)



<!-- ####Examples -->

??? quote "Example"

    Full example for Image Classification Pipeline can be found [HERE](https://colab.research.google.com/drive/1O7op_RtuNs12uIs0QVbwoeZdtbyQ4Q9i#scrollTo=njIH9PnCLhHp)
