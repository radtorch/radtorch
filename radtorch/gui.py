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
import ipywidgets as widgets



def image_classification():
    style={}

    folder = widgets.Text(placeholder='Path to data folder', description='Data Folder:', style=style)
    table = widgets.Text(placeholder='label table: path to csv or name of pandas', description='Label Table:', style=style)
    dicom = widgets.ToggleButtons(options=[True, False],button_style='info',description='DICOM:', style=style,)
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
                        )

    balance_class = widgets.ToggleButtons(options=[True, False],button_style='info',description='Balance:', style=style)
    normalize = widgets.Text(placeholder='place a tuple here to normalize', button_style='info',description='Normalize:', style=style)
    image_resize = widgets.IntText(value=False, description='Custom Resize:', style=style)



    model_type = widgets.Dropdown(options=["vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "wide_resnet50_2", "wide_resnet101_2", "alexnet"],value='vgg16',description='Model Arch:',)
    pre_trained = widgets.ToggleButtons(options=[True, False], value=True, button_style='info',description='Pre Trained:')
    unfreeze = widgets.ToggleButtons(options=[True, False],value=False, button_style='info',description='Unfreeze:')

    classifier_type = widgets.Dropdown(options=["linear_regression", "sgd", "logistic_regression", "ridge", "knn", "decision_trees", "random_forests", "gradient_boost", "adaboost", "xgboost", "nn_classifier"],value='ridge',description='Classifier:', style=style,)
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
        readout_format='.1f', style=style,
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
        readout_format='.1f', style=style,
    )

    cross_validation =widgets.ToggleButtons(options=[True, False],value=True, button_style='info',description='CV:', style=style)
    stratified = widgets.ToggleButtons(options=[True, False],value=True, button_style='info',description='Stratified:', style=style)
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
                        readout_format='d', style=style
                        )
    parameters = widgets.Text(placeholder='Dictionary of Extra classifier parameters', description='Parameters:', style=style)

    # Side Buttons
    save=widgets.Button(
        description='Save Pipeline',
        disabled=False,
        button_style='success', # 'success', 'info', 'warning', 'danger' or ''
        icon='save' # (FontAwesome names without the `fa-` prefix)
    )




    data_entry = widgets.VBox([folder, table, dicom, batch, image_resize, balance_class, normalize, ])
    feature_extraction = widgets.VBox([model_type, pre_trained, unfreeze])
    classifier = widgets.VBox([classifier_type, valid_percent, test_percent, cross_validation, stratified, cross_validation_splits, parameters])
    side_buttons= widgets.VBox([save])

    tab_content = widgets.HBox([data_entry, feature_extraction, classifier, side_buttons])
    # tab = widgets.Tab(children = [tab_content])
    # tab.set_title(0, 'Image Classification')
    # tab

    return tab_content
