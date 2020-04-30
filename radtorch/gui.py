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
from radtorch import pipeline
import ipywidgets as widgets
from IPython.display import clear_output


class Image_Classification():

    def __init__(self, **kwargs):

        # Styles
        style={}
        top_margin=widgets.Layout(margin='10px 0 0 0')

        # Data Module
        self.folder = widgets.Text(placeholder='Path to data folder', description='Data Folder:', style=style)
        # table = widgets.Text(placeholder='label table: path to csv or name of pandas', description='Label Table:', value=None, style=style, layout=top_margin)
        self.dicom = widgets.ToggleButtons(options=[True, False],button_style='',description='DICOM:', value=False, style=style, layout=top_margin)
        self.batch = widgets.IntSlider(
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
        self.balance_class = widgets.ToggleButtons(options=[True, False],button_style='',description='Balance:', style=style, layout=top_margin)
        self.normalize = widgets.ToggleButtons(options=[True, False],button_style='',description='Normalize:', style=style, layout=top_margin)
        self.custom_resize = widgets.IntText(description='Custom Resize:', style=style, value=None, layout=top_margin)

        # Feature Extraction Module
        self.model_arch = widgets.Dropdown(options=["vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "wide_resnet50_2", "wide_resnet101_2", "alexnet"],value='vgg16',description='Model Arch:', layout=top_margin)
        self.pre_trained = widgets.ToggleButtons(options=[True, False], value=True, button_style='',description='Pre Trained:', layout=top_margin)
        self.unfreeze = widgets.ToggleButtons(options=[True, False],value=False, button_style='',description='Unfreeze:', layout=top_margin)

        # Classifier Module
        self.classifier_type = widgets.Dropdown(options=["linear_regression", "sgd", "logistic_regression", "ridge", "knn", "decision_trees", "random_forests", "gradient_boost", "adaboost", "xgboost", "nn_classifier"],value='ridge',description='Classifier:', style=style, layout=top_margin)
        self.valid_percent = widgets.FloatSlider(
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
        self.test_percent = widgets.FloatSlider(
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
        self.cross_validation =widgets.ToggleButtons(options=[True, False],value=True, button_style='',description='CV:', style=style, layout=top_margin)
        self.stratified = widgets.ToggleButtons(options=[True, False],value=True, button_style='',description='Stratified:', style=style, layout=top_margin)
        self.cross_validation_splits = widgets.IntSlider(
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
        self.parameters = widgets.Text(placeholder='Dictionary of Extra classifier parameters', description='Parameters:', style=style, layout=top_margin)



        # Side Buttons
        self.save=widgets.Button(
            description='Save Pipeline',
            disabled=False,
            button_style='success', # 'success', 'info', 'warning', 'danger' or ''
            icon='save' # (FontAwesome names without the `fa-` prefix)
        )
        self.info=widgets.Button(
            description='Show Dataset Information',
            disabled=False,
            button_style='info', # 'success', 'info', 'warning', 'danger' or ''
            icon='save', # (FontAwesome names without the `fa-` prefix),
            layout=widgets.Layout(margin='10px 0 0 0')
        )
        self.run=widgets.Button(
            description='Run Pipeline',
            disabled=False,
            button_style='danger', # 'success', 'info', 'warning', 'danger' or ''
            icon='save' # (FontAwesome names without the `fa-` prefix)
            ,    layout=widgets.Layout(margin='10px 0 0 0')
        )


        self.save.on_click(self.save_clf)
        self.info.on_click(self.info_clf)
        self.run.on_click(self.run_clf)

        # Layout Groups
        # data_entry = widgets.VBox([folder, table, dicom, batch, custom_resize, balance_class, normalize, ])
        data_entry = widgets.VBox([self.folder, self.dicom, self.batch, self.custom_resize, self.balance_class, self.normalize, ])
        feature_extraction = widgets.VBox([self.model_arch, self.pre_trained, self.unfreeze], layout=widgets.Layout(margin='0 0 0 50px'))
        classifier = widgets.VBox([self.classifier_type, self.valid_percent, self.test_percent, self.cross_validation, self.stratified, self.cross_validation_splits, self.parameters], layout=widgets.Layout(margin='0 0 0 50px'))
        side_buttons= widgets.VBox([self.save, self.info, self.run], layout=widgets.Layout(margin='0 0 0 50px'))

        output = widgets.HBox([data_entry, feature_extraction, classifier, side_buttons])

        display (output)

    def save_clf(self, button):
        if self.normalize.value==True:
            n=((0,0,0), (1,1,1))
        else:
            n=False

        clf_kwargs={
        'data_directory':self.folder.value,
        'table':None,
        'is_dicom':self.dicom.value,
        'normalize':n,
        'balance_class':self.self.balance_class.value,
        'batch_size':self.batch.value,
        'model_arch':self.model_arch.value,
        'custom_resize':self.custom_resize.value,
        'pre_trained':self.pre_trained.value,
        'unfreeze':self.unfreeze.value,
        'type':self.classifier_type.value,
        'test_percent':self.test_percent.value,
        'valid_percent':self.valid_percent.value,
        'cv':self.cross_validation.value,
        'stratified':self.stratified.value,
        'num_splits':self.cross_validation_splits.value,
        'parameters':self.parameters.value,
        }
        self.clf = pipeline.Image_Classification(**clf_kwargs)
        return self.clf

    def info_clf(self, button):
        self.clf.data_processor.show_dataset_info()

    def run_clf(self, button):
        self.clf.run()
