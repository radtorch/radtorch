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

from radtorch.settings import *
from radtorch.vis import *
from radtorch.general import *
from radtorch.data import *
from radtorch.core import *
from radtorch.pipeline import *
import ipywidgets as widgets



def image_classification():
    # Styles
    style={}
    top_margin=widgets.Layout(margin='10px 0 0 0')

    # Data Module
    folder = widgets.Text(placeholder='Path to data folder', description='Data Folder:', style=style)
    table = widgets.Text(placeholder='label table: path to csv or name of pandas', description='Label Table:', style=style, layout=top_margin)
    dicom = widgets.ToggleButtons(options=[True, False],button_style='',description='DICOM:', style=style, layout=top_margin)
    batch = widgets.IntSlider(
                        value=16,
                        min=4,
                        max=32,
                        step=1,
                        description='Batch Size:',
                        disabled=False,
                        continuous_update=False,
                        orientation='horizontal',
                        readout=True,
                        readout_format='d'
                        , layout=top_margin
                        )
    balance_class = widgets.ToggleButtons(options=[True, False],button_style='',description='Balance:', style=style, layout=top_margin)
    normalize = widgets.Text(placeholder='place a tuple here to normalize', button_style='',description='Normalize:', style=style, layout=top_margin)
    custom_resize = widgets.IntText(value=False, description='Custom Resize:', style=style, layout=top_margin)

    # Feature Extraction Module
    model_arch = widgets.Dropdown(options=["vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "wide_resnet50_2", "wide_resnet101_2", "alexnet"],value='vgg16',description='Model Arch:', layout=top_margin)
    pre_trained = widgets.ToggleButtons(options=[True, False], value=True, button_style='',description='Pre Trained:', layout=top_margin)
    unfreeze = widgets.ToggleButtons(options=[True, False],value=False, button_style='',description='Unfreeze:', layout=top_margin)

    # Classifier Module
    classifier_type = widgets.Dropdown(options=["linear_regression", "sgd", "logistic_regression", "ridge", "knn", "decision_trees", "random_forests", "gradient_boost", "adaboost", "xgboost", "nn_classifier"],value='ridge',description='Classifier:', style=style, layout=top_margin)
    valid_percent = widgets.FloatSlider(
        value=0.2,
        min=0,
        max=1.0,
        step=0.1,
        description='Valid Percent:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f', style=style, layout=top_margin
    )
    test_percent = widgets.FloatSlider(
        value=0.2,
        min=0,
        max=1.0,
        step=0.1,
        description='Test Percent:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f', style=style, layout=top_margin
    )
    cross_validation =widgets.ToggleButtons(options=[True, False],value=True, button_style='',description='CV:', style=style, layout=top_margin)
    stratified = widgets.ToggleButtons(options=[True, False],value=True, button_style='',description='Stratified:', style=style, layout=top_margin)
    cross_validation_splits = widgets.IntSlider(
                        value=5,
                        min=2,
                        max=20,
                        step=1,
                        description='CV Splits:',
                        disabled=False,
                        continuous_update=False,
                        orientation='horizontal',
                        readout=True,
                        readout_format='d', style=style, layout=top_margin
                        )
    parameters = widgets.Text(placeholder='Dictionary of Extra classifier parameters', description='Parameters:', style=style, layout=top_margin)

    clf_kwargs={
    'data_directory':folder,
    'table':table,
    'is_dicom':dicom,
    'normalize':normalize,
    'balance_class':balance_class,
    'batch_size':batch,
    'model_arch':model_arch,
    'custom_resize':custom_resize,
    'pre_trained':pre_trained,
    'unfreeze':unfreeze,
    'type':classifier_type,
    'test_percent':test_percent,
    'valid_percent':valid_percent,
    'cv':cross_validation,
    'stratified':stratified,
    'num_splits':cross_validation_splits,
    'parameters':parameters,
    }

    # Side Buttons
    save=widgets.Button(
        description='Save Pipeline',
        disabled=False,
        button_style='success', # 'success', 'info', 'warning', 'danger' or ''
        icon='save' # (FontAwesome names without the `fa-` prefix)
    )
    info=widgets.Button(
        description='Show Dataset Information',
        disabled=False,
        button_style='info', # 'success', 'info', 'warning', 'danger' or ''
        icon='save', # (FontAwesome names without the `fa-` prefix),
        layout=widgets.Layout(margin='10px 0 0 0')
    )
    run=widgets.Button(
        description='Run Pipeline',
        disabled=False,
        button_style='danger', # 'success', 'info', 'warning', 'danger' or ''
        icon='save' # (FontAwesome names without the `fa-` prefix)
        ,    layout=widgets.Layout(margin='10px 0 0 0')
    )

    def save_clf():
        clf = Image_Classification(**clf_kwargs)
        return clf
    save.on_click(save_clf)

    # Layout Groups
    data_entry = widgets.VBox([folder, table, dicom, batch, custom_resize, balance_class, normalize, ])
    feature_extraction = widgets.VBox([model_arch, pre_trained, unfreeze], layout=widgets.Layout(margin='0 0 0 50px'))
    classifier = widgets.VBox([classifier_type, valid_percent, test_percent, cross_validation, stratified, cross_validation_splits, parameters], layout=widgets.Layout(margin='0 0 0 50px'))
    side_buttons= widgets.VBox([save, info, run], layout=widgets.Layout(margin='0 0 0 50px'))

    output = widgets.HBox([data_entry, feature_extraction, classifier, side_buttons])


    return output
