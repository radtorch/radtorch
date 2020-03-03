# Pipeline Module <small> radtorch.pipeline </small>
<p style='text-align: justify;'>
Pipelines are probably the most exciting feature of RADTorch tool kit. With few lines of code, the pipeline module allows you to run state-of-the-art image classification algorithms and much more.
</p>




## Image_Classification

      pipeline.Image_Classification(data_directory, transformations='default',
      custom_resize='default', device='default', optimizer='Adam', is_dicom=True,
      label_from_table=False, is_csv=None, table_source=None, path_col='IMAGE_PATH',
      label_col='IMAGE_LABEL', mode='RAW', wl=None, batch_size=16, test_percent=0.2,
      valid_percent=0.2, model_arch='vgg16', pre_trained=True, unfreeze_weights=True,
      train_epochs=20, learning_rate=0.0001, loss_function='CrossEntropyLoss')

!!! quote ""

    The Image Classification pipeline simplifies the process of binary and multi-class image classification into a single line of code.
    Under the hood, the following happens:

    1. The pipeline creates a master dataset from the provided data directory and source of labels/classes either from [folder structre](https://pytorch.org/docs/stable/torchvision/datasets.html#datasetfolder) or pandas/csv table.

    2. Master dataset is subdivided into train, valid and test subsets using the percentages defined by user.

    3. The following transformations are applied on the dataset images:
        1. Resize to the default image size allowed by the model architecture.
        2. Window/Level adjustment according to values specified by user.
        3. Single channel grayscale DICOM images are converted into 3 channel grayscale images to fit into the model.

    3. Selected Model architecture, optimizer, and loss function are downloaded/created.

    4. Model is trained.

    5. Training metrics are saved as training progresses and can be displayed after training is done.

    6. Confusion Matrix and ROC (for binary classification) can be displayed as well (by default, the test subset is used to calculate the confusion matrix and the ROC)

    7. Trained model can be exported to outside file for future use.


####Parameters

!!! quote ""

    **data_directory:**

    - _(str)_ target data directory. ***(Required)***

    **is_dicom:**

    - _(boolean)_ True for DICOM images, False for regular images.(default=True)

    **label_from_table:**

    - _(boolean)_ True if labels are to extracted from table, False if labels are to be extracted from subfolders. (default=False)

    **is_csv:**

    - _(boolean)_ True for csv, False for pandas dataframe.

    **table_source:**

    - _(str or pandas dataframe object)_ source for labelling data.This is path to csv file or name of pandas dataframe if pandas to be used. (default=None).

    **path_col:**

    - _(str)_  name of the column with the image path. (default='IMAGE_PATH')

    **label_col:**

    - _(str)_  name of the label/class column. (default='IMAGE_LABEL')

    **mode:**

    - _(str)_  output mode for DICOM images only where RAW= Raw pixels, HU= Image converted to Hounsefield Units, WIN= 'window' image windowed to certain W and L, MWIN = 'multi-window' converts image to 3 windowed images of different W and L (specified in wl argument) stacked together. (default='RAW')

    **wl:**

    - _(list)_ list of lists of combinations of window level and widths to be used with WIN and MWIN.In the form of : [[Level,Width], [Level,Width],...].  
    - Only 3 combinations are allowed for MWIN (for now). (default=None)

    **transformations:**

    - _(pytorch transforms list)_ pytroch transforms to be performed on the dataset. (default=Convert to tensor)

    **custom_resize:**
    - _(int)_ by default, a radtorch pipeline will resize the input images into the default training model input image size as demosntrated in the table shown in radtorch home page. This default size can be changed here if needed.

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

    - _(boolean)_ if True, all model weights will be retrained. (default=True)

    **train_epochs:**

    - _(int)_ Number of training epochs. (default=20)

    **learning_rate:**

    - _(str)_ training learning rate. (default = 0.0001)

    **loss_function:**

    - _(str)_ training loss function. (default='CrossEntropyLoss')

    **optimizer:**

    - _(str)_ Optimizer to be used during training. (default='Adam')

    **device:**

    - _(str)_ device to be used for training. This can be adjusted to 'cpu' or 'cuda'. If nothing is selected, the pipeline automatically detects if cuda is available and trains on it.


####Methods

!!! quote ""


    **.info()**

    - Display Parameters of the Image Classification Pipeline.

    **.dataset_info()**

    - Display Dataset Information.

    **.sample()**

    - Display sample of the training dataset.

    - Arguments:
        - num_of_images_per_row: _(int)_ number of images per column. (default=5)
        - fig_size: _(tuple)_ figure size. (default=(10,10))
        - show_labels: _(boolean)_ show the image label idx. (default=True)

    **.train()**

    - Train the image classification pipeline.

    - Arguments:
        - verbose: _(boolean)_ Show display progress after each epoch. (default=True)

    **.metrics()**

    - Display the training metrics.

    **.export_model()**

    - Export the trained model into a target file.

    - Arguments:
        - output_path: _(str)_ path to output file. For example 'foler/folder/model.pth'

    **.export()**

    - Exports the whole image classification pipeline for future use

    - Arguments:
        - target_path: _(str)_ target location for export.


    **.set_trained_model()**

    - Loads a previously trained model into pipeline

    - Arguments:
        - model_path: _(str)_ path to target model
        - mode: _(str)_ either 'train' or 'infer'.'train' will load the model to be trained. 'infer' will load the model for inference.


    **.inference()**

    - Performs inference using the trained model on a target image.

    - Arguments:
        - test_img_path: _(str)_ path to target image.
        - transformations: _(pytorch transforms list)_ list of transforms to be performed on the target image. (default='default' which is the same transforms using for training the pipeline)

    - Outputs:
        - Output: _(tuple)_ tuple of prediction (class idx , accuracy percentage).


    **.roc()**

    - Display ROC and AUC.

    - Arguments:
        - target_data_set: _(pytorch dataset object)_ dataset used for predictions to create the ROC. By default, the image classification pipeline uses the test dataset created to calculate the ROC. If no test dataset was created in the pipeline (e.g. test_percent=0), then an external test dataset is required. (default=default')
        - auc: _(boolen)_ Display area under curve. (default=True)
        - figure_size: _(tuple)_ figure size. (default=(7,7))


    **.confusion_matrix()**

    - Display Confusion Matrix

    - Arguments:
        - target_data_set: _(pytorch dataset object)_ dataset used for predictions to create the confusion matrix. By default, the image classification - pipeline uses the test dataset created to calculate the matrix.
        - target_classes: _(list)_ list of classes. By default, the image classification pipeline uses the training classes.
        - figure_size: _(tuple)_ figure size. (default=(7,7))

####Examples

!!! quote ""

    **Importing the pipeline and setting up data directory**

        from radtorch import pipeline
        data_root = '/content/data'


    **Create the image classifier pipeline**

    The below example will create the pipeline using resnet50 model architecture with trained weights loaded and will train for 30 epochs.

        clf = pipeline.Image_Classification(data_directory=data_root,
              mode='HU', model_arch='resnet50', train_epochs=30)

    **Show dataset Information**

        clf.dataset_info()

    <!-- **** -->


        Number of intances = 77
        Number of classes =  2
        Class IDX =  {'axr': 0, 'cxr': 1}

        Class Frequency:
        Class Number of instances
         1       39
         0       38
        None
        Train Dataset Size  47
        Valid Dataset Size  15
        Test Dataset Size  15

    **Display sample of the dataset**

        clf.sample()

    ![](img/sample.png)

    **Train the classifier**

        clf.train()

    <!-- **** -->


        Starting training at 2020-02-27 17:47:54.918173
        Epoch : 000/30 : [Training: Loss: 0.7937, Accuracy: 46.8085%]  [Validation : Loss : 0.7436, Accuracy: 40.0000%] [Time: 1.4200s]
        Epoch : 001/30 : [Training: Loss: 0.6054, Accuracy: 63.8298%]  [Validation : Loss : 0.7793, Accuracy: 40.0000%] [Time: 1.3436s]
        Epoch : 002/30 : [Training: Loss: 0.5504, Accuracy: 80.8511%]  [Validation : Loss : 1.2314, Accuracy: 40.0000%] [Time: 1.3500s]
        ...
        ...
        ...
        Epoch : 026/30 : [Training: Loss: 0.0499, Accuracy: 97.8723%]  [Validation : Loss : 0.4143, Accuracy: 93.3333%] [Time: 1.3612s]
        Epoch : 027/30 : [Training: Loss: 0.0235, Accuracy: 97.8723%]  [Validation : Loss : 0.0321, Accuracy: 100.000%] [Time: 1.3540s]
        Epoch : 028/30 : [Training: Loss: 0.0142, Accuracy: 100.000%]  [Validation : Loss : 0.2476, Accuracy: 93.3333%] [Time: 1.3584s]
        Epoch : 029/30 : [Training: Loss: 0.0067, Accuracy: 100.000%]  [Validation : Loss : 0.4216, Accuracy: 93.3333%] [Time: 1.3728s]

        Total training time = 0:00:40.758802


    **Display training metrics**

        clf.metrics()

    ![](img/metrics.png)


    **Display Confusion Matrix**

        clf.confusion_matrix()

    ![](img/cm.png)


    **Display ROC**

        clf.roc()

    ![](img/roc.png)

    **Export Trained Model**

        clf.export_model('/folder/model.pth')


<hr>



## Feature_Extraction

    pipeline.Feature_Extraction(data_directory, transformations='default',
    custom_resize = 'default', is_dicom=True,label_from_table=False,
    is_csv=None,table_source=None, device='default', path_col = 'IMAGE_PATH',
    label_col = 'IMAGE_LABEL', mode='RAW', wl=None, model_arch='vgg16',
    pre_trained=True, unfreeze_weights=False)

!!! quote ""

    The feature extraction pipeline utilizes a pre-trained model to extract a set of features that can be used in another machine learning algorithms e.g. XGBoost. The trained model by default can one of the supported model architectures trained with default weights trained on the ImageNet dataset or a model that has been trained and exported using the image classification pipeline.

    The output is a pandas dataframe that has feature columns, label column and file path column.

    Under the hood, the pipeline removes the last FC layer of the pretrained models to output the features.

    The number of extracted features depends on the model architecture selected:

    <div align='center'>

    | Model Architecture | Default Input Image Size | Output Features |
    |--------------------|:------------------------:|:---------------:|
    | VGG16              |         244 x 244        |       4096      |
    | VGG19              |         244 x 244        |       4096      |
    | resnet50           |         244 x 244        |       2048      |
    | resnet152          |         244 x 244        |       2048      |
    | resnet101          |         244 x 244        |       2048      |
    | wide_resnet50_2    |         244 x 244        |       2048      |
    | wide_resnet101_2   |         244 x 244        |       2048      |
    | inception_v3       |         299 x 299        |       2048      |

    </div>


####Parameters

!!! quote ""

    **data_directory:**

    - _(str)_ target data directory. ***(Required)***

    **is_dicom:**

    - _(boolean)_  True for DICOM images, False for regular images.(default=True)

    **label_from_table:** [boolean] True if labels are to extracted from table, False if labels are to be extracted from subfolders. (default=False)

    **is_csv:**

    - _(boolean)_  True for csv, False for pandas dataframe.

    **table_source:**

    - _(str or pandas dataframe object)_ source for labelling data. (default=None). This is path to csv file or name of pandas dataframe if pandas to be used.

    **path_col:**

    - _(str)_ name of the column with the image path. (default='IMAGE_PATH')

    **label_col:**

    - _(str)_ name of the label/class column. (default='IMAGE_LABEL')

    **mode:**

    - _(str)_ output mode for DICOM images only.
    - Options:
                   RAW= Raw pixels,
                   HU= Image converted to Hounsefield Units,
                   WIN= 'window' image windowed to certain W and L,
                   MWIN = 'multi-window' converts image to 3 windowed images of different W and L (specified in wl argument) stacked together]. (default='RAW')

    **wl:**

    - _(list)_ list of lists of combinations of window level and widths to be used with WIN and MWIN.
              In the form of : [[Level,Width], [Level,Width],...].
              Only 3 combinations are allowed for MWIN (for now).(default=None)

    **transformations:**

    - _(pytorch transforms)_ pytroch transforms to be performed on the dataset. (default=Convert to tensor)

    **custom_resize:**

    - _(int)_ by default, a radtorch pipeline will resize the input images into the default training model input image
    size as demosntrated in the table shown in radtorch home page. This default size can be changed here if needed.
    model_arch: [str] PyTorch neural network architecture (default='vgg16')

    **pre_trained:**

    - _(boolean)_  Load the pretrained weights of the neural network. If False, the last layer is only retrained = Transfer Learning. (default=True)

    **unfreeze_weights:**

    - _(boolean)_  if True, all model weights, not just final layer, will be retrained. (default=False)

    **device:**

    - _(str)_ device to be used for training. This can be adjusted to 'cpu' or 'cuda'. If nothing is selected, the pipeline automatically detects if cuda is available and trains on it.


####Methods

!!! quote ""


    **.info()**

    - Displays Feature Extraction Pipeline Parameters.

    **.dataset_info()**

    - Display Dataset Information.

    **.sample()**

    - Display sample of the training dataset.

    **.num_features()**

    - Displays number of features to be extracted.

    **.run()**

    - Extracts features from dataset.

    - Arguments:
        - verbose: _(boolean)_ Show the feature table. (default=True)

    **.export_features()**

    - Exports the features to csv.

    - Arguments:
        - csv_path: _(str)_ Path to output csv file.


    **set_trained_model**

    - Loads a previously trained model into pipeline

    - Arguments:
        - model_path: _(str)_ path to target model
        - mode: _(str)_ either 'train' or 'infer'.'train' will load the model to be trained. 'infer' will load the model for inference.

####Examples

!!! quote ""

    **Importing the pipeline and setting up data directory**
    ```
    from radtorch import pipeline
    data_root = '/content/data'

    ```
    **Create the feature extractor pipeline**

    The below example will create the pipeline using resnet152 model architecture with trained weights loaded.

    ```
    extractor = pipeline.Feature_Extraction(data_directory=data_root, mode='HU',
                model_arch='resnet152')
    ```


    **Display number of Features to be extracted**
    ```
    extractor.num_features()
    ```
    ```
    2048
    ```

    **Show Dataset information**
    ```
    extractor.dataset_info()
    ```
    ```
    Number of intances = 77
    Number of classes =  2
    Class IDX =  {'axr': 0, 'cxr': 1}

    Class Frequency:
    Class Number of instances
    0       38
    1       39
    ```

    **Display sample of the dataset**
    ```
    extractor.sample()
    ```
    ![](img/sample.png)

    **Run pipeline to extract features**
    ```
    extractor.run()
    ```
    ```
      |    | img_path       |  label_idx |       f_0 |      f_1 |         f_2 |      f_3 | ........ |
      |---:|:---------------|-----------:|----------:|---------:|------------:|---------:|---------:|
      |  0 | /content/dat...|          0 | 0.135294  | 0.368051 | 0.000352088 | 0.378677 | ........ |
      |  1 | /content/dat...|          0 | 0.0721618 | 0.930238 | 0.0286931   | 0.732228 | ........ |
      |  2 | /content/dat...|          0 | 0.0780637 | 0.432966 | 0.0175741   | 0.685681 | ........ |
      |  3 | /content/dat...|          0 | 0.560777  | 0.449213 | 0.0432512   | 0.432942 | ........ |
      |  4 | /content/dat...|          0 | 0.176524  | 0.669066 | 0.0396659   | 0.273474 | ........ |


    ```

    **Show feature names list**
    ```
    extractor.feature_names
    ```
    ```
    ['f_0','f_1','f_2','f_3','f_4', 'f_5', ... ]
    ```



## load_pipeline
      pipeline.load_pipeline(target_path)

!!! quote ""
    Loads a previously saved pipeline for future use.

    **Arguments**

    - target_path: _(str)_ target path of the target pipeline.

    **Example**

        my_classifier = load_pipeline('/path/to/pipeline.dump')
