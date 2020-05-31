
import streamlit as st
from radtorch.settings import *
from radtorch import pipeline, core, utils
from radtorch.utils import *
import SessionState

h = st.sidebar.markdown('# RADTorch<small> v1.1.3</small>', unsafe_allow_html=True)
app=st.sidebar.selectbox('Select a Pipeline', ('Home', 'Image Classification', 'Generative Adversarial Networks'))




class Image_Classification_Module():

    def __init__(self):
        ## Main Content
        st.markdown('# Image Classification <small>pipeline</small>', unsafe_allow_html=True)
        self.data_directory= st.text_input('Data Directory (required)', value='/Users/elbanan/Projects/alexmed_data')


        ## Controls
        self.save = st.button('SAVE')
        self.sample = st.button('SAMPLE')
        self.run = st.button('RUN')
        self.data_info = st.button('DATASET INFO')
        self.clear = st.button('CLEAR')


        ## Sidebar
        t = st.sidebar.subheader('Data')
        self.is_table=st.sidebar.checkbox('Label from table', value=False)
        if self.is_table:
            self.table=st.sidebar.file_uploader('Label Table (optional)')
            self.image_path_column=st.sidebar.text_input('Image Path Column',value='IMAGE_PATH')
            self.image_label_column=st.sidebar.text_input('Image Label Column',value='IMAGE_LABEL')
            self.is_path=st.sidebar.checkbox('is path', value=True)
        else:
            self.image_path_column = 'IMAGE_PATH'
            self.image_label_column ='IMAGE_LABEL'
            self.is_path = True
            self.table=None
        self.is_dicom=st.sidebar.checkbox('DICOM images', value=False)

        if self.is_dicom:
            self.mode=st.sidebar.selectbox('Image Mode', ('RAW', 'HU', 'WIN', 'MWIN'))
        else:
            self.mode='RAW'

        self.balance_class=st.sidebar.checkbox('Balance Class', value=False)
        if self.balance_class:
            self.balance_class_method=st.sidebar.selectbox('Balance Method', ('upsample', 'downsample')),
        else:
            self.balance_class_method='upsample'

        self.interaction_terms=st.sidebar.checkbox('Interaction Terms', value=False)

        self.normalize = st.sidebar.checkbox('Normalize', value=True)
        if self.normalize:
            self.normalize_mean=st.sidebar.number_input('Norm Mean', min_value=None, max_value=None,  value=0)
            self.normalize_std=st.sidebar.number_input('Norm Standard Deviation', min_value=None, max_value=None, value=1)

        self.sampling = st.sidebar.number_input('Sampling', min_value=0.01, max_value=1.0, value=1.0, step=0.01)
        self.batch_size = st.sidebar.number_input('Batch Size', min_value=1, max_value=None, value=16)
        self.num_workers = st.sidebar.number_input('Workers', min_value=0, max_value=None, value=0)
        self.test_percent = st.sidebar.number_input('Test Percent', min_value=0.01, max_value=0.9, value=0.2, step=0.01)



        st.sidebar.markdown('---', unsafe_allow_html=True)
        t = st.sidebar.subheader('Feature Extractor')
        self.model_arch=st.sidebar.selectbox('Model Architecture', (supported_models), index=4)
        self.pre_trained=st.sidebar.checkbox('Pre-trained', value=True)
        self.unfreeze=st.sidebar.checkbox('Unfreeze All Layers', value=False)

        st.sidebar.markdown('---', unsafe_allow_html=True)
        t = st.sidebar.subheader('Classifier')
        self.type=st.sidebar.selectbox('Classifier Type', (SUPPORTED_CLASSIFIER), index=10)
        if self.type == 'nn_classifier':
            self.valid_percent = st.sidebar.number_input('Valid Percent', min_value=0.01, max_value=0.9, value=0.2, step=0.01)
            self.learning_rate=st.sidebar.number_input('Learning Rate', min_value=None, max_value=None, value=0.0001, step=0.0001, format='%5f')
            self.epochs=st.sidebar.number_input('Epochs', min_value=1, max_value=None, value=10, step=1)
            self.optimizer=st.sidebar.selectbox('Optimizer', (supported_nn_optimizers), index=0)
            self.loss_function=st.sidebar.selectbox('Loss Function', (supported_nn_loss_functions), index=1)
            self.cv=True
            self.stratified=True
            self.num_splits=5
            # self.parameters=''

        else:
            self.cv=st.sidebar.checkbox('Cross Validation', value=True)
            self.stratified=st.sidebar.checkbox('Stratified', value=True)
            self.num_splits=st.sidebar.number_input('CV Splits', min_value=1, max_value=None, value=5, step=1)
            # self.parameters=st.sidebar.text_input('Classifier Parameters',value='')
            self.valid_percent = 0.2
            self.learning_rate=0.0001
            self.epochs=10
            self.optimizer='Adam'
            self.loss_function='CrossEntropyLoss'

        st.sidebar.markdown('---', unsafe_allow_html=True)
        self.device=st.sidebar.selectbox('Device', ('auto', 'cuda', 'cpu'), index=0)

        self.state = SessionState.get(clf=None)

        if self.save: self.state.clf = self.create_classifier()
        if self.data_info: self.show_data_info()
        if self.sample: self.show_sample()
        if self.clear : st.write('')
        if self.run: self.state.clf.run(gui=True)


    def create_classifier(self):
        clf = pipeline.Image_Classification(
                        data_directory=self.data_directory,
                        is_dicom=self.is_dicom,
                        table=self.table,
                        image_path_column=self.image_path_column,
                        image_label_column=self.image_label_column,
                        is_path=self.is_path,
                        mode=self.mode,
                        balance_class=self.balance_class,
                        balance_class_method=self.balance_class_method,
                        interaction_terms=self.interaction_terms,
                        normalize=((self.normalize_mean,self.normalize_mean,self.normalize_mean), (self.normalize_std,self.normalize_std,self.normalize_std)),
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        sampling=self.sampling,
                        test_percent=self.test_percent,
                        valid_percent=self.valid_percent,
                        model_arch=self.model_arch,
                        pre_trained=self.pre_trained,
                        unfreeze=self.unfreeze,
                        type=self.type,
                        cv=self.cv,
                        stratified=self.stratified,
                        num_splits=self.num_splits,
                        # parameters={k:v for k, v in self.parameters.items()},
                        learning_rate=self.learning_rate,
                        epochs=self.epochs,
                        optimizer=self.optimizer,
                        loss_function=self.loss_function,
                        device=self.device)
        return clf

    def show_data_info(self):
        clf=self.state.clf
        info_dict={}
        info_dict['dataset']=show_dataset_info(clf.data_processor.master_dataset)
        info_dict['dataset'].style.set_caption('Overall Dataset')
        if 'type' in clf.data_processor.__dict__.keys():
            for i in ['train_dataset','test_dataset']:
                if i in clf.data_processor.__dict__.keys():
                    info_dict[i]= show_dataset_info(clf.data_processor.__dict__[i])
                    info_dict[i].style.set_caption(i)
            if clf.data_processor.type=='nn_classifier':
                if 'valid_dataset' in clf.data_processor.__dict__.keys():
                    info_dict['valid_dataset']= show_dataset_info(clf.data_processor.__dict__['valid_dataset'])
                    info_dict[i].style.set_caption('valid_dataset')
            for k, v in info_dict.items():
                print (k)
                st.write(k)
                st.dataframe(v)

    def show_sample(self, figure_size=(10,10)):
        clf=self.state.clf
        clf.data_processor.sample(figure_size=figure_size, gui=True)


if app == 'Image Classification':
    Image_Classification_Module()
# /Users/elbanan/Projects/alexmed_data
