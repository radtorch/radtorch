# Copyright (C) 2020 RADTOrch and Mohamed Elbanan, MD
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see https://www.gnu.org/licenses/

from ..settings import *
from .general import *
from .dicom import *
from .data import *


def plot_images(images, titles=None, figure_size=(10,10)):
    """
    Source
    ---------
    https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
    """
    cols = int(math.sqrt(len(images)))
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure(figsize=figure_size)
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        plt.axis('off')
        a.set_title(title)
    plt.axis('off')
    plt.show()


def misclassified(true_labels_list, predicted_labels_list, accuracy_list, img_path_list):
    misclassified = {}
    for i in range (len(true_labels_list)):
        if true_labels_list[i] != predicted_labels_list[i]:
            misclassified[img_path_list[i]] = {'image_path': img_path_list[i],
                                                'true_label': true_labels_list[i],
                                                'predicted_label': predicted_labels_list[i],
                                                'accuracy':accuracy_list[i]
                                                }
    return misclassified


def show_misclassified(misclassified_dictionary, transforms, class_to_idx_dict, is_dicom = True, num_of_images = 16, figure_size = (5,5), ):
    row = int(math.sqrt(num_of_images))
    try:
        sample = random.sample(list(misclassified_dictionary), num_of_images)
        if is_dicom:
            imgs = [torch.from_numpy(dicom_to_narray(i)) for i in sample]
        else:
            imgs = [np.array(transforms(Image.open(i).convert('RGB'))) for i in sample]
            imgs = [np.moveaxis(i, 0, -1) for i in imgs]

        titles = [
                    ', '.join([
                    ['Truth: '+k for k,v in class_to_idx_dict.items() if v == misclassified_dictionary[i]['true_label']][0],
                    ['Pred: '+k for k,v in class_to_idx_dict.items() if v == misclassified_dictionary[i]['predicted_label']][0],
                    'Acc = '+str(float('{:0.2f}'.format(misclassified_dictionary[i]['accuracy'])))
                    ])

                   for i in sample]

        plot_images(images=imgs, titles=titles, figure_size=figure_size)
    except:
        log("Error! Number of misclassified images is less than 16. Please use a smaller num_of_images to display.")
        pass


def show_dataloader_sample(dataloader, figure_size=(10,10), show_labels=True, show_file_name = False,):
  batch = next(iter(dataloader))
  images, labels, paths = batch
  images = images.numpy()
  images = [np.moveaxis(x, 0, -1) for x in images]
  if show_labels:
      titles = labels.numpy()
      titles = [((list(dataloader.dataset.class_to_idx.keys())[list(dataloader.dataset.class_to_idx.values()).index(i)]), i) for i in titles]
  if show_file_name:
      titles = [ntpath.basename(x) for x in paths]
  plot_images(images=images, titles=titles, figure_size=figure_size)


def show_dataset_info(dataset):
    input_data = dataset.input_data
    image_path_col = dataset.image_path_column
    image_label_col = dataset.image_label_column

    class_names = list(dataset.class_to_idx.keys())+['Total Instances']
    class_idx = list(dataset.class_to_idx.values())+['']
    num_instances = []
    for i in list(dataset.class_to_idx.keys()):
      num_instances.append(input_data[image_label_col].value_counts()[[i]].sum())
    num_instances.append(sum(num_instances))
    output = pd.DataFrame(list(zip(class_names, class_idx, num_instances)), columns=['Classes', 'Class Idx', 'Number of Instances'])
    return output


def show_dicom_sample(dataloader, figsize=(30,10)):
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


def show_nn_misclassified(model, target_data_set, num_of_images, device, transforms, is_dicom = True, figure_size=(5,5)):

    class_dictionary = target_data_set.class_to_idx

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
            softmax = torch.exp(out).cpu()
            accuracies = [(max(i.tolist())) for i in softmax]
            misses = misclassified(true_labels_list=labels.tolist(), predicted_labels_list=pr, img_path_list=list(paths), accuracy_list=accuracies)
            misses_all.update(misses)
            pred_labels = pred_labels+pr

    show_misclassified(misclassified_dictionary=misses_all, transforms=transforms, class_to_idx_dict=class_dictionary, is_dicom = is_dicom, num_of_images = num_of_images, figure_size = figure_size)

    output = pd.DataFrame(misses_all.values())

    return output


def plot_features(feature_table, feature_names, num_features, num_images,image_path_col, image_label_col):

    output_notebook()

    colors = ['#F2F4F4', '#93D5ED', '#45A5F5', '#4285F4', '#2F5EC4', '#0D47A1']
    TOOLS = "hover,save,box_zoom,reset,wheel_zoom, box_select"

    min_fig_size = 400

    f = (feature_table).copy()

    max_value = max(f[feature_names].max().tolist())
    min_value = min(f[feature_names].min().tolist())


    file_label_dict = {}

    for i in sorted(f[image_label_col].unique()):
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
        mapper = LinearColorMapper(palette=colors, low=min_value, high=max_value)

        plot_width = num_features*8
        plot_height=num_images*8

        if plot_width < min_fig_size:
            plot_width = min_fig_size

        if plot_height < min_fig_size:
            plot_height = min_fig_size

        p = figure(title=("Extracted Imaging Features for class "+str(k)),
                x_range=features, y_range=images,
                x_axis_location="above", plot_width=plot_width, plot_height=plot_height,
                tools=TOOLS, toolbar_location='below',
                tooltips=[('instance', '@IMAGE_PATH'), ('feature', '@features'), ('value', '@value')])

        p.grid.grid_line_color = None
        p.axis.axis_line_color = None
        p.axis.major_tick_line_color = None
        p.axis.major_label_text_font_size = "4pt"
        p.axis.major_label_standoff = 0
        p.xaxis.major_label_orientation = pi / 3
        p.toolbar.autohide = True

        p.rect(x="features", y="IMAGE_PATH", width=1, height=1,
            source=df,
            fill_color={'field': 'value', 'transform': mapper},
            line_color=None)

        color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="8pt",
                          ticker=BasicTicker(desired_num_ticks=len(colors)),
                          label_standoff=6, border_line_color=None, location=(0, 0))


        p.add_layout(color_bar, 'right')
        tab = Panel(child=p,title=("Class "+str(k)) )
        figures.append(tab)

    tabs = Tabs(tabs=figures)

    show(tabs)


def plot_pipline_dataset_info(dataframe, test_percent):

    colors = ['#93D5ED', '#45A5F5', '#4285F4', '#2F5EC4', '#0D47A1']
    TOOLS = "hover,save,box_zoom,reset,wheel_zoom, box_select"

    G = dataframe[['Classes', 'Number of Instances']]
    G.columns = ['Classes', 'Number']

    if test_percent==0:
        Z = -3
    else:
        Z = -4

    output_notebook()

    source = ColumnDataSource(G)

    output = []

    p = figure(plot_width=600, plot_height=400, x_range=G['Classes'].tolist()[:Z], tools=TOOLS, tooltips=[('','@Classes'), ('','@Number')], title='Data Breakdown by Class')
    p.vbar(x='Classes', width=0.4, top = 'Number', line_color=None, source=source, fill_color=factor_cmap('Classes', palette=colors[::-1], factors=(G['Classes'].tolist()[:Z])))
    output.append(p)
    p.xaxis.axis_line_color = '#D6DBDF'
    p.yaxis.axis_line_color = '#D6DBDF'
    p.xgrid.grid_line_color=None
    p.yaxis.axis_line_width = 2
    p.xaxis.axis_line_width = 2
    p.xaxis.major_tick_line_color = '#D6DBDF'
    p.yaxis.major_tick_line_color = '#D6DBDF'
    p.xaxis.minor_tick_line_color = '#D6DBDF'
    p.yaxis.minor_tick_line_color = '#D6DBDF'
    p.yaxis.major_tick_line_width = 2
    p.xaxis.major_tick_line_width = 2
    p.yaxis.minor_tick_line_width = 0
    p.xaxis.minor_tick_line_width = 0
    p.xaxis.major_label_text_color = '#99A3A4'
    p.yaxis.major_label_text_color = '#99A3A4'
    p.outline_line_color = None
    p.toolbar.autohide = True



    p = figure(plot_width=600, plot_height=400, x_range=G['Classes'].tolist()[Z:], tooltips=[('','@Classes'), ('','@Number')], title='Data Breakdown by Subsets')
    p.vbar(x='Classes', width=0.5, top = 'Number', line_color=None, source=source, fill_color=factor_cmap('Classes', palette=colors[::-1], factors=(G['Classes'].tolist()[Z:])))
    p.xaxis.axis_line_color = '#D6DBDF'
    p.yaxis.axis_line_color = '#D6DBDF'
    p.xgrid.grid_line_color=None
    p.yaxis.axis_line_width = 2
    p.xaxis.axis_line_width = 2
    p.xaxis.major_tick_line_color = '#D6DBDF'
    p.yaxis.major_tick_line_color = '#D6DBDF'
    p.xaxis.minor_tick_line_color = '#D6DBDF'
    p.yaxis.minor_tick_line_color = '#D6DBDF'
    p.yaxis.major_tick_line_width = 2
    p.xaxis.major_tick_line_width = 2
    p.yaxis.minor_tick_line_width = 0
    p.xaxis.minor_tick_line_width = 0
    p.xaxis.major_label_text_color = '#99A3A4'
    p.yaxis.major_label_text_color = '#99A3A4'
    p.outline_line_color = None
    p.toolbar.autohide = True
    output.append(p)

    show(row(output))


def plot_dataset_info(dataframe_dictionary, plot_size=(500,300)):
    output_notebook()
    output = []
    for dataframe_title , dataframe in dataframe_dictionary.items():
        G = dataframe[['Classes', 'Number of Instances']]
        G.columns = ['Classes', 'Number']
        source = ColumnDataSource(G)
        p = figure(plot_width=plot_size[0], plot_height=plot_size[1], x_range=G['Classes'].tolist(), tools=TOOLS, tooltips=[('','@Classes'), ('','@Number')], title=('Data Breakdown for '+dataframe_title))
        p.vbar(x='Classes', width=0.4, top = 'Number', line_color=None, source=source, fill_color=factor_cmap('Classes', palette=COLORS[::-1], factors=(G['Classes'].tolist())))
        p.xaxis.axis_line_color = '#D6DBDF'
        p.yaxis.axis_line_color = '#D6DBDF'
        p.xgrid.grid_line_color=None
        p.yaxis.axis_line_width = 2
        p.xaxis.axis_line_width = 2
        p.xaxis.major_tick_line_color = '#D6DBDF'
        p.yaxis.major_tick_line_color = '#D6DBDF'
        p.xaxis.minor_tick_line_color = '#D6DBDF'
        p.yaxis.minor_tick_line_color = '#D6DBDF'
        p.yaxis.major_tick_line_width = 2
        p.xaxis.major_tick_line_width = 2
        p.yaxis.minor_tick_line_width = 0
        p.xaxis.minor_tick_line_width = 0
        p.xaxis.major_label_text_color = '#99A3A4'
        p.yaxis.major_label_text_color = '#99A3A4'
        p.outline_line_color = None
        p.toolbar.autohide = True
        output.append(p)

    show(column(output))


def show_metrics(classifer_list, figure_size=(700,400)):
    metrics_list = [x.train_metrics for x in classifer_list]
    output_notebook()
    output = []
    for m in ['Accuracy', 'Loss',]:
        ind = 0
        if m =='Loss':
          legend_items = []
          p = figure(plot_width=figure_size[0], plot_height=figure_size[1], title=('Loss'), tools=TOOLS, toolbar_location='below', tooltips=[('','@x'), ('','@y')])
          for i in metrics_list:
            x = p.line(i.index.to_list(), i.Train_Loss.to_list() , line_width=2, line_color= COLORS2[ind])
            y = p.line(i.index.to_list(), i.Valid_Loss.to_list() , line_width=2, line_color= COLORS2[-ind], line_dash='dotted')
            legend_items.append((('Model '+str(ind)+' Train Loss') , [x]))
            legend_items.append(('Model '+str(ind)+' Valid Loss' , [y]))
            ind = ind +1

        elif m == "Accuracy":
          legend_items = []
          p = figure(plot_width=figure_size[0], plot_height=figure_size[1], title=('Accuracy'), tools=TOOLS, toolbar_location='below', tooltips=[('','@x'), ('','@y')])
          for i in metrics_list:
            x = p.line(i.index.to_list(), i.Train_Accuracy.to_list() , line_width=2, line_color= COLORS2[ind])
            y = p.line(i.index.to_list(), i.Valid_Accuracy.to_list() , line_width=2, line_color= COLORS2[-ind], line_dash='dotted')
            legend_items.append((('Model '+str(ind)+' Train Accuracy') , [x]))
            legend_items.append(('Model '+str(ind)+' Valid Accuracy' , [y]))
            ind = ind +1

        legend = Legend(items=legend_items, location=(10, -20))
        p.add_layout(legend, 'right')
        # p.legend.location = "top_center"
        p.legend.inactive_fill_alpha = 0.7
        p.legend.border_line_width = 0
        p.legend.click_policy="hide"
        p.xaxis.axis_line_color = '#D6DBDF'
        p.yaxis.axis_line_color = '#D6DBDF'
        p.xgrid.grid_line_color=None
        p.yaxis.axis_line_width = 2
        p.xaxis.axis_line_width = 2
        p.xaxis.major_tick_line_color = '#D6DBDF'
        p.yaxis.major_tick_line_color = '#D6DBDF'
        p.xaxis.minor_tick_line_color = '#D6DBDF'
        p.yaxis.minor_tick_line_color = '#D6DBDF'
        p.yaxis.major_tick_line_width = 2
        p.xaxis.major_tick_line_width = 2
        p.yaxis.minor_tick_line_width = 0
        p.xaxis.minor_tick_line_width = 0
        p.xaxis.major_label_text_color = '#99A3A4'
        p.yaxis.major_label_text_color = '#99A3A4'
        p.outline_line_color = None
        p.xaxis.axis_label = 'Epoch'
        p.xaxis.axis_label_text_align = 'right'
        p.toolbar.autohide = True
        output.append(p)


    show(column(output))


def calculate_nn_predictions(model, target_data_set,  device):
    """
    """

    true_labels = []
    pred_labels = []
    model.to(device)
    target_data_loader = torch.utils.data.DataLoader(target_data_set,batch_size=16,shuffle=False)

    for i, (imgs, labels, path) in tqdm(enumerate(target_data_loader), total=len(target_data_loader)):
        imgs = imgs.to(device)
        labels = labels.to(device)
        true_labels = true_labels+labels.tolist()
        with torch.no_grad():
            model.eval()
            out = model(imgs)
            ps = out
            pr = [(i.tolist()).index(max(i.tolist())) for i in ps]
            pred_labels = pred_labels+pr

    return (true_labels, pred_labels)


def show_roc(classifier_list, figure_size=(700,400)):

    output_notebook()

    TOOLS = "hover,save,box_zoom,reset,wheel_zoom, box_select"

    output = []
    p = figure(plot_width=figure_size[0], plot_height=figure_size[1], title=('Receiver Operating Characteristic'), tools=TOOLS, toolbar_location='below', tooltips=[('','@x'), ('','@y')])
    p.line([0, 0.5, 1.0], [0, 0.5, 1.0], line_width=1.5, line_color='#93D5ED', line_dash='dashed')

    ind = 0

    auc_list = []

    legend_items = []

    for i in classifier_list:
        if i.type in [x for x in SUPPORTED_CLASSIFIER if x != 'nn_classifier']:
            true_labels=i.test_labels
            predictions=i.classifier.predict(i.test_features)
        else: true_labels, predictions = calculate_nn_predictions(model=i.trained_model, target_data_set=i.test_dataset, device=i.device)
        fpr, tpr, thresholds = metrics.roc_curve(true_labels, predictions)
        auc = metrics.roc_auc_score(true_labels, predictions)
        x = p.line(fpr, tpr, line_width=2, line_color= COLORS2[ind])
        legend_items.append((('Model '+i.classifier_type+'. AUC = '+'{:0.4f}'.format((auc))),[x]))

        ind = ind+1
        auc_list.append(auc)

    legend = Legend(items=legend_items, location=(10, -20))
    p.add_layout(legend, 'right')

    p.legend.inactive_fill_alpha = 0.7
    p.legend.border_line_width = 0
    p.legend.click_policy="hide"
    p.xaxis.axis_line_color = '#D6DBDF'
    p.xaxis.axis_label = 'False Positive Rate (1-Specificity)'
    p.yaxis.axis_label = 'True Positive Rate (Senstivity)'
    p.yaxis.axis_line_color = '#D6DBDF'
    p.xgrid.grid_line_color=None
    p.yaxis.axis_line_width = 2
    p.xaxis.axis_line_width = 2
    p.xaxis.major_tick_line_color = '#D6DBDF'
    p.yaxis.major_tick_line_color = '#D6DBDF'
    p.xaxis.minor_tick_line_color = '#D6DBDF'
    p.yaxis.minor_tick_line_color = '#D6DBDF'
    p.yaxis.major_tick_line_width = 2
    p.xaxis.major_tick_line_width = 2
    p.yaxis.minor_tick_line_width = 0
    p.xaxis.minor_tick_line_width = 0
    p.xaxis.major_label_text_color = '#99A3A4'
    p.yaxis.major_label_text_color = '#99A3A4'
    p.outline_line_color = None
    p.toolbar.autohide = True

    show(p)

    return auc_list
