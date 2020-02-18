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




def show_dataloader_sample(dataloader, num_of_images_per_row=10, figsize=(10,10), show_labels=True):
  """
    Displays sample of certain dataloader with corresponding class idx
    Inputs:
        dataloader: [dataloader object] selected pytorch dataloader
        num_of_images_per_row: [int] number of images per row (default=)
        figsize: [tuple] size of displayed figure (default = (10,10))
        show_labels: [boolen] display class idx of the sample displayed (default=True)

    .. image:: pass.jpg
  """

  batch = next(iter(dataloader))
  images, labels = batch
  grid = torchvision.utils.make_grid(images, nrow=num_of_images_per_row)
  plt.figure(figsize=(figsize))
  plt.imshow(np.transpose(grid, (1,2,0)))
  if show_labels:
      print ('labels:', labels)

def show_roc(true_labels, predicted_labels, auc = True, figsize=(10,10), title='ROC Curve'):
    """
    Displays ROC curve of a certain model and the AUC
    Inputs:
        true_labels: [list] true labels of test set
        predicted_labels: [list] predicted labels of test set using target model
        auc: [boolen] displays area under curve for ROC (default=True)
        figsize: [tuple] size of displayed figure (default = (10,10))
        title: [str] title of the displayed curve (default = 'ROC Curve')

    .. image:: pass.jpg
    """

    fpr, tpr, thresholds = metrics.roc_curve(true_labels, predicted_labels)
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr)
    plt.title(title);
    plt.xlabel('FPR (1-specficity)');
    plt.ylabel('TPR (Sensitivity)');
    plt.grid(True)
    if auc:
        print (metrics.roc_auc_score(true_labels, predicted_labels))

def show_dataset_info(dataset):
    """
    Displays a summary of the pytorch dataset information
    Inputs:
        dataset: [pytorch dataset object] target dataset.

    .. image:: pass.jpg
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

def show_metrics(source, fig_size=(15,5)):
    """
    Displays metrics created by the training loop

    .. image:: pass.jpg    
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
    Displays an sample image from a dataloader.
    Returns a single image in case of one window and 3 images in case of mutiple window.
    Inputs:
        dataloader: [pytorch dataloader object] target dataloader
        figsize: [tuple] size of displayed figure when 3 images are displayed (default = (30,10))

    .. image:: pass.jpg
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
