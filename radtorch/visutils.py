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


from radtorch.generalutils import getDuplicatesWithCount




def show_dataloader_sample(dataloader, num_of_images_per_row=10, figsize=(10,10), show_labels=False):
  """
    Displays sample of certain dataloader with corresponding class idx

    **Arguments**

    - dataloader: _(dataloader object)_ selected pytorch dataloader.

    - num_of_images_per_row: _(int)_ number of images per row. (default=10)

    - figsize: _(tuple)_ size of displayed figure. (default = (10,10))

    - show_labels: _(boolen)_ display class idx of the sample displayed .(default=False)

    **Output**

    -  Output: _(figure)_
  """

  batch = next(iter(dataloader))
  images, labels, paths = batch
  grid = torchvision.utils.make_grid(images, nrow=num_of_images_per_row)
  plt.figure(figsize=(figsize))
  plt.imshow(np.transpose(grid, (1,2,0)))
  if show_labels:
      print ('labels:', labels)

def show_dataset_info(dataset):
    """
    Displays a summary of the pytorch dataset information.

    **Arguments**

    - dataset: _(pytorch dataset object)_ target dataset to inspect.

    **Output**

    -  Output: _(str)_ Dataset information including:
        - Number of instances
        - Number of classes
        - Dictionary of class and class_id
        - Class frequency breakdown.
    """

    label_list = [i[1] for i in dataset]
    label_stats = getDuplicatesWithCount(label_list)

    print ('Number of intances =', len(dataset))
    print ('Number of classes = ', len(dataset.classes))
    print ('Class IDX = ', dataset.class_to_idx)
    print ('')
    print ('Class Frequency: ')
    print ('{0:2s} {1:3s}'.format('Class', 'Number of instances'))
    for key, value in label_stats.items():
        print('{0:2d} {1:3s} {2:4d}'.format(key, '',value))

    class_names = list(dataset.class_to_idx.keys())+['Total Instances']
    class_idx = list(dataset.class_to_idx.values())
    num_instances = list(label_stats.values())+[len(dataset)]
    output = pd.DataFrame([class_names, class_idx, num_instances], columns=['Classes', 'Class Idx', 'Number of Instances'])
    return output

def show_metrics(source, fig_size=(15,5)):
    """
    Displays metrics created by the training loop.

    **Arguments**

    - source: _(list)_ the metrics generated during the training process as by modelsutils.train_model()

    - fig_size: _(tuple)_ size of the displayed figure. (default=15,5)

    **Output**

    -  Output: _(figure)_ Matplotlib graphs of accuracy and error for training and validation.
    """

    metrics = np.array(source)
    loss = metrics[:,0:2]
    accuracy = metrics[:,2:4]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)

    ax1.plot(loss)
    ax1.legend(['Train Loss', 'Valid Loss'])
    ax1.set(xlabel='Epoch Number', ylabel='Loss')
    ax1.grid(True)
    ax2.plot(accuracy)
    ax2.legend(['Train Accuracy', 'Valid Accuracy'])
    ax2.set(xlabel='Epoch Number', ylabel='Accuracy')
    ax2.grid(True)

def show_dicom_sample(dataloader, figsize=(30,10)):
    """
    Displays a sample image from a DICOM dataloader. Returns a single image in case of one window and 3 images in case of multiple window.

    **Arguments**

    - dataloader: _(dataloader object)_ selected pytorch dataloader.

    - figsize: _(tuple)_ size of the displayed figure. (default=30,10)

    **Output**

    -  Output: _(figure)_ single image in case of one window and 3 images in case of multiple window.
    """

    i, l = next(iter(dataloader))
    l = l.detach().cpu().numpy();
    if i[0].shape[0] == 3:
        x, ([ax1, ax2, ax3]) = plt.subplots(1,3, figsize=figsize);
        ax1.imshow(i[0][0], cmap='gray');
        ax1.set_title(l[0]);
        ax1.axis('off');
        ax2.imshow(i[0][1], cmap='gray');
        ax2.set_title(l[0]);
        ax2.axis('off');
        ax3.imshow(i[0][2], cmap='gray');
        ax3.set_title(l[0]);
        ax3.axis('off');
    else:
        plt.imshow(i[0][0], cmap='gray');
        plt.title(l[0]);

def show_roc(true_labels, predictions, auc=True, figure_size=(10,10), title='ROC Curve'):
    """
    Displays ROC curve and AUC using true and predicted label lists.

    **Arguments**

    - true_labels: _(list)_ list of true labels.

    - predictions: _(list)_ list of predicted labels.

    - auc: _(boolen)_ True to display AUC. (default=True)

    - figure_size: _(tuple)_ size of the displayed figure. (default=10,10)

    - title: _(str)_ title displayed on top of the output figure. (default='ROC Curve')

    **Output**

    -  Output: _(figure)_
    """
    fpr, tpr, thresholds = metrics.roc_curve(true_labels, predictions)
    plt.figure(figsize=figure_size)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='orange', alpha=.8)
    plt.plot(fpr, tpr)
    plt.title(title);
    plt.xlabel('FPR (1-specficity)');
    plt.ylabel('TPR (Sensitivity)');
    plt.grid(True)
    if auc == True:
        plt.xlabel('FPR (1-specficity)\nAUC={:0.4f}'.format(metrics.roc_auc_score(true_labels, predictions)))
        # print ('AUC =',metrics.roc_auc_score(true_labels, predictions))
        return metrics.roc_auc_score(true_labels, predictions)

def show_nn_roc(model, target_data_set,  device, auc=True, figure_size=(10,10)):
    """
    Displays the ROC and AUC of a certain trained model on a target(for example test) dataset.

    **Arguments**

    - model: _(pytorch model object)_ target model.

    - target_data_set: _(pytorch dataset object)_ target dataset.

    - auc: _(boolen)_ True to display AUC. (default=True)

    - figure_size: _(tuple)_ size of the displayed figure. (default=10,10)

    - device: _(str)_ device for inference. 'cpu' or 'cuda'


    **Output**

    -  Output: _(figure)_

    """

    true_labels = []
    pred_labels = []
    model.to(device)
    target_data_loader = torch.utils.data.DataLoader(target_data_set,batch_size=16,shuffle=False)

    for i, (imgs, labels, path) in tqdm(enumerate(target_data_loader), total=len(target_data_loader)):
        imgs = imgs.to(device)
        labels = labels.to(device)
        true_labels = true_labels+labels.tolist()
        # print (imgs.shape)
        with torch.no_grad():
            model.eval()
            out = model(imgs)
            # ps = torch.exp(out)
            ps = out
            pr = [(i.tolist()).index(max(i.tolist())) for i in ps]
            pred_labels = pred_labels+pr


    show_roc(true_labels, pred_labels, auc=auc, figure_size=figure_size)

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion Matrix',
                          cmap=None,
                          normalize=False,
                          figure_size=(8,6)):
    """
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

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=figure_size)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

def show_confusion_matrix(model, target_data_set, target_classes, device, figure_size=(8,6), cmap=None):
    '''
    Displays Confusion Matrix for Image Classifier Model.

    **Arguments**

    - model: _(pytorch model object)_ target model.

    - target_data_set: _(pytorch dataset object)_ target dataset.

    - target_classes: _(list)_ list of class names.

    - figure_size: _(tuple)_ size of the displayed figure. (default=8,6)

    - cmap: _(str)_ the colormap of the generated figure (default=None, which is Blues)

    - device: _(str)_ device for inference. 'cpu' or 'cuda'

    **Output**

    -  Output: _(figure)_
    '''
    true_labels = []
    pred_labels = []
    model.to(device)
    target_data_loader = torch.utils.data.DataLoader(target_data_set,batch_size=16,shuffle=False)

    for i, (imgs, labels, path) in tqdm(enumerate(target_data_loader), total=len(target_data_loader)):
        imgs = imgs.to(device)
        labels = labels.to(device)
        true_labels = true_labels+labels.tolist()
        # print (imgs.shape)
        with torch.no_grad():
            model.eval()
            out = model(imgs)
            # ps = torch.exp(out)
            ps = out
            pr = [(i.tolist()).index(max(i.tolist())) for i in ps]
            pred_labels = pred_labels+pr


    cm = metrics.confusion_matrix(true_labels, pred_labels)
    plot_confusion_matrix(cm=cm,
                          target_names=target_classes,
                          title='Confusion Matrix',
                          cmap=cmap,
                          normalize=False,
                          figure_size=figure_size
                          )
