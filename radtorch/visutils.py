import torch, torchvision, datetime, time, pickle, pydicom, os, math, random, itertools
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

from bokeh.io import output_notebook
from math import pi
from bokeh.io import show
from bokeh.models import BasicTicker, ColorBar, LinearColorMapper, PrintfTickFormatter, Tabs, Panel
from bokeh.plotting import figure
from bokeh.sampledata.unemployment1948 import data
from bokeh.layouts import row, gridplot


from radtorch.generalutils import getDuplicatesWithCount
from radtorch.dicomutils import dicom_to_narray




def misclassified(true_labels_list, predicted_labels_list, img_path_list):
    misclassified = {}
    for i in range (len(true_labels_list)):
        if true_labels_list[i] != predicted_labels_list[i]:
            misclassified[img_path_list[i]] = {'image_path': img_path_list[i], 'true_label': true_labels_list[i], 'predicted_label': predicted_labels_list[i]}
    return misclassified


def show_misclassified(misclassified_dictionary, is_dicom = True, num_of_images = 16, figure_size = (5,5)):
    row = int(math.sqrt(num_of_images))
    sample = random.sample(list(misclassified_dictionary), num_of_images)
    transform=transforms.Compose([transforms.Resize((244, 244)),transforms.ToTensor()])
    if is_dicom:
        imgs = [torch.from_numpy(dicom_to_narray(i)) for i in sample]
    else:
        imgs = [transform(Image.open(i).convert('RGB')) for i in sample]
    grid = torchvision.utils.make_grid(imgs, nrow=row)
    plt.figure(figsize=(figure_size))
    plt.imshow(np.transpose(grid, (1,2,0)))


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

    input_data = dataset.input_data
    image_path_col = dataset.image_path_col
    image_label_col = dataset.image_label_col



    class_names = list(dataset.class_to_idx.keys())+['Total Instances']
    class_idx = list(dataset.class_to_idx.values())+['']
    num_instances = []
    for i in list(dataset.class_to_idx.keys()):
      num_instances.append(input_data[image_label_col].value_counts()[[i]].sum())
    num_instances =num_instances+[len(dataset)]
    output = pd.DataFrame(list(zip(class_names, class_idx, num_instances)), columns=['Classes', 'Class Idx', 'Number of Instances'])

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


def show_confusion_matrix(cm,target_names,title='Confusion Matrix',cmap=None,normalize=False,figure_size=(8,6)):
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


def show_nn_confusion_matrix(model, target_data_set, target_classes, device, figure_size=(8,6), cmap=None):
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

    for i, (imgs, labels, paths) in tqdm(enumerate(target_data_loader), total=len(target_data_loader)):
        imgs = imgs.to(device)
        labels = labels.to(device)
        true_labels = true_labels+labels.tolist()
        with torch.no_grad():
            model.eval()
            out = model(imgs)
            ps = out
            pr = [(i.tolist()).index(max(i.tolist())) for i in ps]
            pred_labels = pred_labels+pr


    cm = metrics.confusion_matrix(true_labels, pred_labels)
    show_confusion_matrix(cm=cm,
                          target_names=target_classes,
                          title='Confusion Matrix',
                          cmap=cmap,
                          normalize=False,
                          figure_size=figure_size
                          )


def show_nn_misclassified(model, target_data_set, num_of_images, device, is_dicom = True, figure_size=(5,5)):
    true_labels = []
    pred_labels = []
    misses_all = {}

    model.to(device)
    target_data_loader = torch.utils.data.DataLoader(target_data_set,batch_size=16,shuffle=False)

    for i, (imgs, labels, paths) in tqdm(enumerate(target_data_loader), total=len(target_data_loader)):
        imgs = imgs.to(device)
        labels = labels.to(device)
        true_labels = true_labels+labels.tolist()
        with torch.no_grad():
            model.eval()
            out = model(imgs)
            ps = out
            pr = [(i.tolist()).index(max(i.tolist())) for i in ps]
            misses = misclassified(true_labels_list=labels.tolist(), predicted_labels_list=pr, img_path_list=list(paths))
            # misses_all = dict(misses_all.items() + misses.items())
            misses_all.update(misses)
            pred_labels = pred_labels+pr

    show_misclassified(misclassified_dictionary=misses_all, is_dicom = is_dicom, num_of_images = num_of_images, figure_size = figure_size)
    output = pd.DataFrame(misses_all.values())

    return output


def plot_features(feature_table, feature_names, num_features, num_images,image_path_col, image_label_col, split_by_class=False):
    '''
    .. include:: ./documentation/docs/visutils.md##plot_features
    '''

    output_notebook()

    colors = ['#F2F4F4', '#93D5ED', '#45A5F5', '#4285F4', '#2F5EC4', '#0D47A1']
    TOOLS = "hover,save,box_zoom,reset,wheel_zoom, box_select"


    f = (feature_table).copy()


    file_label_dict = {}

    for i in f[image_label_col].unique():
        file_label_dict[str(i)] = f[f[image_label_col] == i];

    figures = []

    for k, v in file_label_dict.items():
        f = v[:num_images]
        f = f[[image_path_col]+feature_names[:num_features]]
        f[image_path_col] = f[image_path_col].astype(str)
        i = f[image_path_col].tolist()
        i = [os.path.basename(str(x)) for x in i]
        f[image_path_col] = i

        f = f.set_index(image_path_col)


        f.columns.name = 'features'
        images = list(f.index)
        features = list(f.columns)

        df = pd.DataFrame(f.stack(), columns=['value']).reset_index()
        mapper = LinearColorMapper(palette=colors, low=df.value.min(), high=df.value.max())

        p = figure(title=("Extracted Imaging Features for class "+str(k)),
                x_range=features, y_range=images,
                x_axis_location="above", plot_width=num_features*8, plot_height=num_images*8,
                tools=TOOLS, toolbar_location='below',
                tooltips=[('image', '@img_path'), ('feature', '@features'), ('value', '@value')])

        p.grid.grid_line_color = None
        p.axis.axis_line_color = None
        p.axis.major_tick_line_color = None
        p.axis.major_label_text_font_size = "4pt"
        p.axis.major_label_standoff = 0
        p.xaxis.major_label_orientation = pi / 3

        p.rect(x="features", y="img_path", width=1, height=1,
            source=df,
            fill_color={'field': 'value', 'transform': mapper},
            line_color=None)

        color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="8pt",
                          ticker=BasicTicker(desired_num_ticks=len(colors)),
                          #  formatter=PrintfTickFormatter(format="%d%%"),
                          label_standoff=6, border_line_color=None, location=(0, 0))


        p.add_layout(color_bar, 'right')
        tab = Panel(child=p,title=("Class "+str(k)) )
        figures.append(tab)

        # show(p)
    tabs = Tabs(tabs=figures)

    show(tabs)
