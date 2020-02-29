# Visualization Module <small> radtorch.visutils </small>

Different tools and utilities for data visualization. Based upon Matplotlib.

    from radtorch import visutils

##show_dataloader_sample
    visutils.show_dataloader_sample(dataloader, num_of_images_per_row=10,
                            figsize=(10,10), show_labels=True)

!!! quote ""
    Displays sample of certain dataloader with corresponding class idx

    **Arguments**

    - dataloader: _(dataloader object)_ selected pytorch dataloader.

    - num_of_images_per_row: _(int)_ number of images per row. (default=10)

    - figsize: _(tuple)_ size of displayed figure. (default = (10,10))

    - show_labels: _(boolen)_ display class idx of the sample displayed .(default=True)

    **Output**

    -  Output: _(figure)_



##show_dataset_info
    visutils.show_dataset_info(dataset)

!!! quote ""
    Displays a summary of the pytorch dataset information.

    **Arguments**

    - dataset: _(pytorch dataset object)_ target dataset to inspect.

    **Output**

    -  Output: _(str)_ Dataset information including:
        - Number of instances
        - Number of classes
        - Dictionary of class and class_id
        - Class frequency breakdown.



##show_metrics
    visutils.show_metrics(source, fig_size=(15,5))

!!! quote ""
    Displays metrics created by the training loop.

    **Arguments**

    - source: _(list)_ the metrics generated during the training process as by modelsutils.train_model()

    - fig_size: _(tuple)_ size of the displayed figure. (default=15,5)

    **Output**

    -  Output: _(figure)_ Matplotlib graphs of accuracy and error for training and validation.



##show_dicom_sample
    visutils.how_dicom_sample(dataloader, figsize=(30,10))

!!! quote ""
    Displays a sample image from a DICOM dataloader. Returns a single image in case of one window and 3 images in case of multiple window.

    **Arguments**

    - dataloader: _(dataloader object)_ selected pytorch dataloader.

    - figsize: _(tuple)_ size of the displayed figure. (default=30,10)

    **Output**

    -  Output: _(figure)_ single image in case of one window and 3 images in case of multiple window.



##show_roc
    visutils.show_roc(true_labels, predictions, auc=True,
                      figure_size=(10,10), title='ROC Curve')

!!! quote ""
    Displays ROC curve and AUC using true and predicted label lists.

    **Arguments**

    - true_labels: _(list)_ list of true labels.

    - predictions: _(list)_ list of predicted labels.

    - auc: _(boolen)_ True to display AUC. (default=True)

    - figure_size: _(tuple)_ size of the displayed figure. (default=10,10)

    - title: _(str)_ title displayed on top of the output figure. (default='ROC Curve')

    **Output**

    -  Output: _(figure)_



##show_nn_roc
    visutils.show_nn_roc(model, target_data_set, auc=True, figure_size=(10,10))

!!! quote ""
    Displays the ROC and AUC of a certain trained model on a target(for example test) dataset.

    **Arguments**

    - model: _(pytorch model object)_ target model.

    - target_data_set: _(pytorch dataset object)_ target dataset.

    - auc: _(boolen)_ True to display AUC. (default=True)

    - figure_size: _(tuple)_ size of the displayed figure. (default=10,10)

    **Output**

    -  Output: _(figure)_



##plot_confusion_matrix
    visutils.plot_confusion_matrix(cm,target_names, title='Confusion Matrix',
                                  cmap=None,normalize=False,figure_size=(8,6))

!!! quote ""
    Given a sklearn confusion matrix (cm), make a nice plot. Code adapted from : https://www.kaggle.com/grfiv4/plot-a-confusion-matrix.

    **Arguments**

    - cm: _(numpy array)_ confusion matrix from sklearn.metrics.confusion_matrix.

    - target_names: _(list)_ list of class names.

    - title: _(str)_ title displayed on top of the output figure. (default='Confusion Matrix')

    - cmap: _(str)_ The gradient of the values displayed from matplotlib.pyplot.cm . See http://matplotlib.org/examples/color/colormaps_reference.html. (default=None which is plt.get_cmap('jet') or plt.cm.Blues)

    - normalize: _(boolean)_  If False, plot the raw numbers. If True, plot the proportions. (default=False)

    - figure_size: _(tuple)_ size of the displayed figure. (default=8,6)

    **Output**

    -  Output: _(figure)_



##show_confusion_matrix
    visutils.show_confusion_matrix(model, target_data_set,
              target_classes, figure_size=(8,6), cmap=None)

!!! quote ""
    Displays Confusion Matrix for Image Classifier Model.

    **Arguments**

    - model: _(pytorch model object)_ target model.

    - target_data_set: _(pytorch dataset object)_ target dataset.

    - target_classes: _(list)_ list of class names.

    - figure_size: _(tuple)_ size of the displayed figure. (default=8,6)

    - cmap: _(str)_ the colormap of the generated figure (default=None, which is Blues)

    **Output**

    -  Output: _(figure)_
