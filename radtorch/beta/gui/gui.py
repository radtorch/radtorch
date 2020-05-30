import streamlit as st
import pandas as pd
import time

from radtorch.settings import *
from radtorch import pipeline, core, utils


## Main Content
st.markdown('# RADTorch Image Classification <small>v1.1.3</small>', unsafe_allow_html=True)
st.info('Welcome to RADTorch Image Classification Pipeline')



data_directory= st.text_input('Data Directory (required)')
table=st.file_uploader('Label Table (optional)')
save = st.button('SAVE')
sample = st.button('SAMPLE')
data_info = st.button('DATASET INFO')
run = st.button('RUN')



## Side Bar
t = st.sidebar.header('Hyperparameters')
st.sidebar.markdown('---', unsafe_allow_html=True)
t = st.sidebar.subheader('Data')

if table:
    image_path_column=st.sidebar.text_input('Image Path Column',value='IMAGE_PATH')
    image_label_column=st.sidebar.text_input('Image Label Column',value='IMAGE_LABEL')
    is_path=st.sidebar.checkbox('is path', value=True)
else:
    image_path_column = 'IMAGE_PATH'
    image_label_column ='IMAGE_LABEL'
    is_path = True

is_dicom=st.sidebar.checkbox('DICOM images', value=False)

if is_dicom:
    mode=st.sidebar.selectbox('Image Mode', ('RAW', 'HU', 'WIN', 'MWIN'))
else:
    mode='RAW'


balance_class=st.sidebar.checkbox('Balance Class', value=False)
if balance_class:
    balance_class_method=st.sidebar.selectbox('Balance Method', ('upsample', 'downsample')),
else:
    balance_class_method='upsample'

interaction_terms=st.sidebar.checkbox('Interaction Terms', value=False)

normalize = st.sidebar.checkbox('Normalize', value=False)
if normalize:
    normalize_mean=st.sidebar.number_input('Norm Mean', min_value=None, max_value=None,  value=0)
    normalize_std=st.sidebar.number_input('Norm Standard Deviation', min_value=None, max_value=None, value=1)

sampling = st.sidebar.number_input('Sampling', min_value=0.01, max_value=1.0, value=1.0, step=0.01)
batch_size = st.sidebar.number_input('Batch Size', min_value=1, max_value=None, value=16)
num_workers = st.sidebar.number_input('Workers', min_value=0, max_value=None, value=0)
test_percent = st.sidebar.number_input('Test Percent', min_value=0.01, max_value=0.9, value=0.2, step=0.01)


st.sidebar.markdown('---', unsafe_allow_html=True)
t = st.sidebar.subheader('Feature Extractor')
model_arch=st.sidebar.selectbox('Model Architecture', (supported_models), index=4)
pre_trained=st.sidebar.checkbox('Pre-trained', value=True)
unfreeze=st.sidebar.checkbox('Unfreeze All Layers', value=False)

st.sidebar.markdown('---', unsafe_allow_html=True)
t = st.sidebar.subheader('Classifier')
type=st.sidebar.selectbox('Classifier Type', (SUPPORTED_CLASSIFIER), index=10)
if type == 'nn_classifier':
    valid_percent = st.sidebar.number_input('Valid Percent', min_value=0.01, max_value=0.9, value=0.2, step=0.01)
    learning_rate=st.sidebar.number_input('Learning Rate', min_value=None, max_value=None, value=0.0001, step=0.0001, format='%5f')
    epochs=st.sidebar.number_input('Epochs', min_value=1, max_value=None, value=10, step=1)
    optimizer=st.sidebar.selectbox('Optimizer', (supported_nn_optimizers), index=0)
    loss_function=st.sidebar.selectbox('Loss Function', (supported_nn_loss_functions), index=1)
    cv=True
    stratified=True
    num_splits=5
    parameters=''

else:
    cv=st.sidebar.checkbox('Cross Validation', value=True)
    stratified=st.sidebar.checkbox('Stratified', value=True)
    num_splits=st.sidebar.number_input('CV Splits', min_value=1, max_value=None, value=5, step=1)
    parameters=st.sidebar.text_input('Classifier Parameters',value='')
    valid_percent = 0.2
    learning_rate=0.0001
    epochs=10
    optimizer='Adam'
    loss_function='CrossEntropyLoss'

st.sidebar.markdown('---', unsafe_allow_html=True)
device=st.sidebar.selectbox('Device', ('auto', 'cuda', 'cpu'), index=0)


## Controls
if save:
    clf = pipeline.Image_Classification(
                    data_directory=data_directory,
                    is_dicom=is_dicom,
                    table=table,
                    image_path_column=image_path_column,
                    image_label_column=image_label_column,
                    is_path=is_path,
                    mode=mode,
                    # wl=None,
                    balance_class=balance_class,
                    balance_class_method=balance_class_method,
                    interaction_terms=interaction_terms,
                    normalize=((normalize_mean,normalize_mean,normalize_mean), (normalize_std,normalize_std,normalize_std)),
                    batch_size=batch_size,
                    num_workers=num_workers,
                    sampling=sampling,
                    test_percent=test_percent,
                    valid_percent=valid_percent,
                    # custom_resize=False,
                    model_arch=model_arch,
                    pre_trained=pre_trained,
                    unfreeze=unfreeze,
                    type=type,
                    cv=cv,
                    stratified=stratified,
                    num_splits=num_splits,
                    parameters={parameters},
                    learning_rate=learning_rate,
                    epochs=epochs,
                    optimizer=optimizer,
                    loss_function=loss_function,
                    # lr_scheduler=None,
                    # custom_nn_classifier=None,
                    # loss_function_parameters={},
                    # optimizer_parameters={},
                    # transformations='default',
                    # extra_transformations=None,
                    device=device)
    clf.data_processor.dataset_info(plot=False)



    if data_info:
        x = clf.data_processor.table
        print (x)
        st.dataframe(x)
