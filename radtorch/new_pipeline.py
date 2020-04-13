from radtorch.settings import *
from radtorch.model import *
from radtorch.data import *
from radtorch.vis import *
from radtorch.general import *
from radtorch.dataset import *
from radtorch.core import *

#device, table, data_directory, is_dicom, normalize, balance_class, batch_size, num_workers, model_arch , custom_resize, pre_trained, unfreeze, classifier_type, 'test_percent', 'cv', 'stratified', 'num_splits', 'label_column', 'parameters'
DEF = {
'table':None,
'is_dicom':True,
'normalize':((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
'balance_class':False,
'batch_size':16,
'num_workers':1,
'model_arch':'vgg16',
'custom_resize':False,
'pre_trained':True,
'unfreeze':False,
'classifier_type':'ridge',
'test_percent':0.2,
'cv':True,
'stratified':True,
'num_splits':5,
'label_column':'label_idx',
'parameters':{}
}


class Image_Classification():

    def __init__(**kwargs):

        for k, v in kwargs.items():
            setattr(self, k, v)
        for k, v  in DEF.items():
            if k not in kwargs.keys():
                setattr(self, k, v)

        if 'device' not in kwargs.keys(): self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data_processor_params={k:v for k,v in self.__dict__.items() if k in ['device', 'table', 'data_directory', 'is_dicom', 'normalize', 'balance_class', 'batch_size', 'num_workers', 'model_arch' , 'custom_resize']}
        self.feature_extractor_params={k:v for k,v in self.__dict__.items() if k in ['model_arch', 'pre_trained', 'unfreeze']}
        self.classifier_param={k:v for k,v in self.__dict__.items() if k in ['classifier_type', 'test_percent', 'cv', 'stratified', 'num_splits', 'label_column', 'parameters']}

        self.data_processor=Data_Preprocessor(**self.data_processor_params)
        self.feature_extractor=Feature_Extractor(dataloader=self.data_processor.dataloader, **self.feature_extractor_params)

    def info(self):
        info=pd.DataFrame.from_dict(({key:str(value) for key, value in self.__dict__.items()}).items())
        info.columns=['Property', 'Value']
        return info

    def sample(self, **kw): #figure_size=(10,10), show_labels=True, show_file_name=False
        self.data_processor.sample(**kw)

    def run(self, **kw):
        if 'feature_table' in kw.keys():
            self.feature_table=kw['feature_table']
            self.feature_names=kw['feature_names']
        else:
            self.feature_extractor.run()
            self.feature_table=self.feature_extractor.feature_table
            self.feature_names=self.feature_extractor.feature_names
        self.classifier=Classifier(feature_table=self.feature_table, feature_names=self.feature_names, **self.classifier_param)
        self.classfier.run()
        self.trained_model=self.classifier
        self.train_metrics=self.classfier.training_metrics
        self.feature_selector=Feature_Selection(type=self.classifier.type, feature_table=self.feature_extractor.feature_table, feature_names=self.feature_extractor.feature_names, **kwargs)

    def metrics(self, figure_size=(500,300)):
        return show_metrics(self.classifiers,  fig_size=figure_size)

    def export(self, output_path):
        try:
            outfile=open(output_path,'wb')
            pickle.dump(self,outfile)
            outfile.close()
            print ('Pipeline exported successfully.')
        except:
            raise TypeError('Error! Pipeline could not be exported.')
