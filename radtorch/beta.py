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
from ipywidgets import interact, interact_manual
from IPython.display import clear_output


class Image_Classification_UI():

  def __init__(self):

    # Visual Styles
    top_margin=widgets.Layout(margin='10px 0 0 0')
    left_margin=widgets.Layout(margin='0 0 0 10px')
    style = {'description_width': 'initial'}

    # Header and Titles
    self.header = widgets.HTML(description='',value='<h1> <font color="#566573">RAD</font><font color="#ef6603">Torch</font><font color="#566573"> Image Classification Pipeline</font> </h1> ')
    self.title1 = widgets.HTML(description='',value='<h2> <font color="#85929E">Data</font></h2>')
    self.title2 = widgets.HTML(description='',value='<h2> <font color="#85929E">Feature Extractor</font></h2>')
    self.title3 = widgets.HTML(description='',value='<h2> <font color="#85929E">Classifier</font></h2>', layout=widgets.Layout(margin='0 0 0 30px'))
    self.title4 = widgets.HTML(description='',value='<h2> <font color="#85929E">Classifier Parameters</font></h2>')
    self.title5 = widgets.HTML(description='',value='<h3> <font color="#AEB6BF">ML CLassifiers Parameters</font></h3>')
    self.title6 = widgets.HTML(description='',value='<h3> <font color="#AEB6BF">NN Classifiers Parameters</font></h3>', layout=widgets.Layout(margin='0 0 0 180px'))


    # UI Elements
    ## Data
    self.ui_folder = widgets.Text(placeholder='Path to data folder', description='Data Folder:', value='/content/alexmed_data')
    self.ui_table = widgets.Text(placeholder='label table', description='Label Table:')
    self.ui_dicom = widgets.Dropdown(description='DICOM', options=[True, False], value=False)
    self.ui_mode = widgets.Dropdown(options=['RAW', 'HU', 'WIN', 'MWIN'], value='RAW', description='Mode:', )
    self.ui_wl = widgets.Text(placeholder='Window/Level', description='W/L:', value='',)
    self.ui_normalize = widgets.Dropdown(options=[False, ((0,0,0), (1,1,1)), ((0.5,0.5,0.5), (0.5,0.5,0.5))], value=False, description='Normalize:')
    self.ui_balance_class = widgets.Dropdown(description='Balance:', options=[True, False], value=False,layout=left_margin)

    ## Feature Extractor
    self.ui_models = widgets.Dropdown(options=model_dict.keys(), value='vgg16', description='Model Arch:')

    ## Classifier
    self.ui_classifier_type=widgets.Dropdown(options=SUPPORTED_CLASSIFIER, value='logistic_regression', description='Type:', )

    ### ML Classifier
    self.ui_cv=widgets.Dropdown(description='CrossValidation:', options=[True, False], value=True, layout=top_margin)
    self.ui_stratified=widgets.Dropdown(description='Stratified:', options=[True, False], value=True, layout=top_margin)
    self.ui_num_splits=widgets.IntSlider(value=5,min=2,max=20,step=1,description='Splits:',layout=top_margin)
    self.ui_test_percent=widgets.FloatSlider(value=0.2,min=0.1,max=0.8,step=0.1,description='Test %:',layout=top_margin)


    ### NN Classifier
    self.ui_batch_size=widgets.IntSlider(value=16,min=1,max=128,step=1,description='Batch:',layout=top_margin)
    self.ui_workers=widgets.IntSlider(value=0,min=0,max=100,step=1,description='Workers:',layout=top_margin)
    self.ui_unfreeze=widgets.Dropdown(description='Unfreeze:', options=[True, False], value=False, layout=top_margin)
    self.ui_valid_percent=widgets.FloatSlider(value=0.2,min=0.1,max=0.8,step=0.1,description='Valid %:',layout=top_margin)
    self.ui_learning_rate=widgets.FloatText(value=0.00001,description='LR:',layout=top_margin)
    self.ui_epochs=widgets.IntSlider(value=5,min=1,max=1000,step=1,description='Epochs:',layout=top_margin)
    self.ui_optimizer=widgets.Dropdown(description='Optimizer:', options=supported_nn_optimizers, value='Adam', layout=top_margin)
    self.ui_loss_function=widgets.Dropdown(description='Loss F:', options=supported_nn_loss_functions, value='CrossEntropyLoss', layout=top_margin)




    ## Controls
    self.b1=widgets.Button(description='CREATE / SAVE',button_style='success')
    self.b2=widgets.Button(description='CLEAR OUTPUT',button_style='warning')
    self.b3=widgets.Button(description='SAMPLE',button_style='info')
    self.b4=widgets.Button(description='RUN',button_style='danger')
    self.b5=widgets.Button(description='INFO',button_style='info')
    self.b6=widgets.Button(description='DATA INFO',button_style='info')
    self.b7=widgets.Button(description='ROC',button_style='info')
    self.b8=widgets.Button(description='CONFUSION MATRIX',button_style='info')
    self.b9=widgets.Button(description='TEST ACC',button_style='info')
    self.b10=widgets.Button(description='METRICS',button_style='info')


    # UI Layouts
    self.ui1a = widgets.HBox([self.ui_folder,  self.ui_table, self.ui_normalize, self.ui_balance_class],)
    self.ui1b = widgets.HBox([self.ui_dicom, self.ui_mode, self.ui_wl],layout=top_margin)
    self.ui2a = widgets.VBox([self.title2,self.ui_models,])
    self.ui2b = widgets.VBox([self.title3, self.ui_classifier_type])
    self.ui2 = widgets.HBox([self.ui2a, self.ui2b])
    self.ui3 = widgets.HBox([self.title5, self.title6])
    self.ui4a = widgets.VBox([self.ui_cv, self.ui_stratified, self.ui_num_splits, self.ui_test_percent])
    self.ui4b = widgets.VBox([self.ui_batch_size, self.ui_workers,self.ui_valid_percent, self.ui_epochs,  ])
    self.ui4c = widgets.VBox([self.ui_learning_rate, self.ui_unfreeze, self.ui_optimizer, self.ui_loss_function])
    self.ui4 = widgets.HBox([self.ui4a,  self.ui4c, self.ui4b,])
    # self.ui_controls = widgets.HBox([self.b1, self.b5, self.b6, self.b3, self.b4, self.b2, self.b7,self.b8,self.b9], layout=widgets.Layout(margin='40px 0 0 40px'))

    self.ui_c1=widgets.HBox([widgets.VBox([self.b1, self.b3]), widgets.VBox([self.b5, self.b6])],layout=widgets.Layout(margin='40px 0 40px 40px'))
    self.ui_c2=widgets.HBox([widgets.VBox([self.b4, self.b10])],layout=widgets.Layout(margin='40px 0 40px 40px'))
    self.ui_c3=widgets.HBox([widgets.VBox([self.b8, self.b7]), widgets.VBox([self.b9, self.b2])],layout=widgets.Layout(margin='40px 0 40px 40px'))
    self.ui_controls = widgets.HBox([self.ui_c1, self.ui_c2, self.ui_c3])

    self.ui = widgets.VBox([self.header, self.title1, self.ui1a, self.ui1b, self.ui2,self.ui3, self.ui4, self.ui_controls])


    # Control Actions
    self.b1.on_click(self.create_clf)
    self.b2.on_click(self.clear)
    self.b3.on_click(self.sample)
    self.b4.on_click(self.run_clf)
    self.b5.on_click(self.info)
    self.b6.on_click(self.data_info)
    self.b7.on_click(self.roc)
    self.b8.on_click(self.cm)
    self.b9.on_click(self.test_accuracy)
    self.b10.on_click(self.metrics)

    # Display Layouts
    self.output = widgets.Output()
    display(self.ui)
    display(self.output)

  def create_clf(self, button):
    self.output.clear_output()
    if len(self.ui_table.value)==0:
      self.table = None

    with self.output:
      self.clf = Image_Classification(data_directory=self.ui_folder.value,
                                               is_dicom=self.ui_dicom.value,
                                               table=self.table,
                                               mode=self.ui_mode.value,
                                               wl=self.ui_wl.value,
                                               balance_class=self.ui_balance_class.value,
                                               normalize=self.ui_normalize.value,
                                               model_arch=self.ui_models.value,
                                               type=self.ui_classifier_type.value,
                                               cv = self.ui_cv.value,
                                               stratified = self.ui_stratified.value,
                                               num_splits = self.ui_num_splits.value,
                                               test_percent = self.ui_test_percent.value,
                                               batch_size = self.ui_batch_size.value,
                                               num_workers = self.ui_workers.value,
                                               valid_percent =  self.ui_valid_percent.value,
                                               epochs = self.ui_epochs.value,
                                               learning_rate = self.ui_learning_rate.value,
                                               unfreeze = self.ui_unfreeze.value,
                                               optimizer =  self.ui_optimizer.value,
                                               loss_function = self.ui_loss_function.value

                                               )

  def clear(self, button):
    self.output.clear_output()

  def sample(self, button):
    self.output.clear_output()
    with self.output:
      self.clf.data_processor.sample()

  def run_clf(self, button):
    self.output.clear_output()
    with self.output:
      self.clf.run()

  def data_info(self, button):
    self.output.clear_output()
    with self.output:
      self.clf.data_processor.dataset_info(plot=False)

  def info(self, button):
    self.output.clear_output()
    with self.output:
      display(self.clf.info())

  def roc(self, button):
    self.output.clear_output()
    with self.output:
      self.clf.classifier.roc()

  def cm(self, button):
    self.output.clear_output()
    with self.output:
      self.clf.classifier.confusion_matrix()

  def test_accuracy(self, button):
    self.output.clear_output()
    with self.output:
      display(self.clf.classifier.test_accuracy())
                             
  def metrics(self, button):
    self.output.clear_output()
    with self.output:
      show_metrics([self.clf.classifier])                            
