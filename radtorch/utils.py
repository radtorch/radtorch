# Copyright (C) 2020 RADTorch and Mohamed Elbanan, MD
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see https://www.gnu.org/licenses/

# Documentation update: 5/8/2020

from radtorch.settings import *




def root_to_class(root):

    """
    Description
    -----------
    Creates a list and dictionary of classes from folder/subfolder structure

    Parameters
    -----------
    root (string, required): path of data folder.

    Returns
    ----------
    classes (list): list of classes created from subfolders.
    class_to_idx (dictionary): dictionary of mapping of classes to class idx.

    """

    classes = [d.name for d in os.scandir(root) if d.is_dir()]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx  # QC


def list_of_files(root):

    """
    Description
    -----------
    Creates a list of all files in a folder/subfolder.

    Parameters
    -----------
    root (string, required): path of data folder.

    Returns
    ----------
    list of file paths in all folders/subfolders in root.

    """

    listOfFile = os.listdir(root)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(root, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + list_of_files(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles


def path_to_class(filepath):

    """
    Description
    -----------
    Creates a class name from a file path parent folder name.

    Parameters
    -----------
    filepath (string, required): path of target file.

    Returns
    ----------
    String with class name.

    """

    item_class = (Path(filepath)).parts
    return item_class[-2]


def create_data_table(directory, is_dicom, image_path_column, image_label_column):

    """
    Description
    -----------
    Creates a pandas table of image path and image label from target folder.

    Parameters
    -----------
    directory (string, required): path of target data folder.
    is_dicom (bolean, required): True if images are DICOM.
    image_path_column (string, required): name of image path column in output data table.
    image_label_column (string, required):name of image label column in output data table.

    Returns
    ----------
    Pandas dataframe with 2 columns:  image_path_column and image_label_column

    """

    classes, class_to_idx=root_to_class(directory)
    all_files=list_of_files(directory)
    if is_dicom:
        dataset_files=[x for x in all_files  if x.endswith('.dcm')]
    else: dataset_files=[x for x in all_files if x.endswith(IMG_EXTENSIONS)]
    all_classes=[path_to_class(i) for i in dataset_files]
    table=pd.DataFrame(list(zip(dataset_files, all_classes)), columns=[image_path_column, image_label_column])
    return table


def balance_dataset(dataset, label_col, method='upsample'):

    """
    Description
    -----------
    Creates a new RADTorch dataset in which all classes are balanced. This is done using upsampling or downsampling.

    Parameters
    -----------
    dataset (RADTorch dataset object, required): target dataset.
    label_col (string, required): name of the table column containing the labels.
    method (string, requried): methodology used to balance classes. Options={'upsample', 'downsample'}. default='upsample'.

    Returns
    -----------
    RADTorch dataset in which all classes are balanced

    """

    balanced_dataset=copy.deepcopy(dataset)
    df = balanced_dataset.input_data
    counts=df.groupby(label_col).count()
    classes=df[label_col].unique().tolist()
    max_class_num=counts.max()[0]
    max_class_id=counts.idxmax()[0]
    min_class_num=counts.min()[0]
    min_class_id=counts.idxmin()[0]
    if method=='upsample':
        resampled_subsets = [df[df[label_col]==max_class_id]]
        for i in [x for x in classes if x != max_class_id]:
          class_subset=df[df[label_col]==i]
          upsampled_subset=resample(class_subset, n_samples=max_class_num, random_state=100)
          resampled_subsets.append(upsampled_subset)
    elif method=='downsample':
        resampled_subsets = [df[df[label_col]==min_class_id]]
        for i in [x for x in classes if x != min_class_id]:
          class_subset=df[df[label_col]==i]
          upsampled_subset=resample(class_subset, n_samples=min_class_num, random_state=100)
          resampled_subsets.append(upsampled_subset)
    resampled_df = pd.concat(resampled_subsets)
    balanced_dataset.input_data=resampled_df
    return balanced_dataset


def show_dataset_info(dataset):

    """
    Description
    -----------
    Creates table with dataset information and class breakdown.

    Parameters
    -----------
    dataset (RADTorch dataset object, required): target dataset.

    Returns
    -----------
    pandas table with dataset information and class breakdown.

    """

    class_names = list(dataset.class_to_idx.keys())+['Total Instances']
    class_idx = list(dataset.class_to_idx.values())+['']
    num_instances = []
    for i in list(dataset.class_to_idx.keys()):
      num_instances.append(dataset.input_data[dataset.image_label_column].value_counts()[[i]].sum())
    num_instances.append(sum(num_instances))
    output = pd.DataFrame(list(zip(class_names, class_idx, num_instances)), columns=['Classes', 'Class Idx', 'Number of Instances'])
    return output


def plot_dataset_info(dataframe_dictionary, plot_size=(500,300)):

    """
    Description
    -----------
    Displays graphical information of the datasets and the class breakdown. Datasets must be supplied in dictionary format as {'train': train dataset, 'valid': valid dataset, 'test': test dataset}.

    Parameters
    -----------
    dataframe_dictionary (dictionary, required): Datasets supplied in dictionary format as {'train': train dataset, 'valid': valid dataset, 'test': test dataset}.
    plot_size (tuple, optional): tuple of plot size as width and lenght. default=(500,300)

    Returns
    -----------
    Bokeh Graph.

    """

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


def plot_features(feature_table, feature_names, num_features, num_images,image_path_col, image_label_col):
    """

    Description
    -----------
    Displays extracted imaging features as heatmap using Bokeh.


    Parameters
    -----------

    - feature_table (pandas dataframe, required): table containing extracted features and labels.

    - feature_names (list, required): list of feature names.

    - num_features (integer, required): number of features to display.

    - num_images (integer, required): number of images to display features for.

    - image_path_col (string, required): name of column that has image names/path.

    - image_label_col (string, required): name of column that has image labels.


    """

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
