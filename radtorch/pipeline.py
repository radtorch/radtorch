from radtorch.settings import *
from radtorch.model import *
from radtorch.vis import *
from radtorch.general import *
from radtorch.dataset import *
from radtorch.core import *

#device, table, data_directory, is_dicom, normalize, balance_class, batch_size, num_workers, model_arch , custom_resize, pre_trained, unfreeze, classifier_type, 'test_percent', 'cv', 'stratified', 'num_splits', 'label_column', 'parameters'

class Image_Classification():

    def __init__(self, DEFAULT_SETTINGS=IMAGE_CLASSIFICATION_PIPELINE_SETTINGS, **kwargs):

        for k, v in kwargs.items():
            setattr(self, k, v)
        for k, v in DEFAULT_SETTINGS.items():
            if k not in kwargs.keys():
                setattr(self, k, v)

        if 'device' not in kwargs.keys(): self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_processor=Data_Processor(**self.__dict__)
        self.feature_extractor=Feature_Extractor(dataloader=self.data_processor.dataloader, **self.__dict__)

    def info(self):
        info=pd.DataFrame.from_dict(({key:str(value) for key, value in self.__dict__.items()}).items())
        info.columns=['Property', 'Value']
        return info

    def run(self, **kw):
        if 'feature_table' in kw.keys():
            print ('Loading Extracted Features')
            self.feature_table=kw['feature_table']
            self.feature_names=kw['feature_names']
        elif 'feature_table' not in self.__dict__.keys():
            print ('Running Feature Extraction.')
            self.feature_extractor.run()
            self.feature_table=self.feature_extractor.feature_table
            self.feature_names=self.feature_extractor.feature_names
        self.classifier=Classifier(**self.__dict__)
        print ('Running Classifier Training.')
        self.classifier.run()
        self.trained_model=self.classifier
        self.train_metrics=self.classifier.train_metrics
        self.feature_selector=Feature_Selection(type=self.classifier.type, feature_table=self.feature_extractor.feature_table, feature_names=self.feature_extractor.feature_names)
        print ('Classifier Training completed successfully.')

    def metrics(self, figure_size=(500,300)):
        return show_metrics([self.classifier],  fig_size=figure_size)

    def export(self, output_path):
        try:
            outfile=open(output_path,'wb')
            pickle.dump(self,outfile)
            outfile.close()
            print ('Pipeline exported successfully.')
        except:
            raise TypeError('Error! Pipeline could not be exported.')


class Image_Classifier_Selection()

    def __init__(self, DEFAULT_SETTINGS=IMAGE_CLASSIFICATION_PIPELINE_SETTINGS, **kwargs):

        for k, v in kwargs.items():
            setattr(self, k, v)
        for k, v in DEFAULT_SETTINGS.items():
            if k not in kwargs.keys():
                setattr(self, k, v)

        self.compare_parameters={k:v for k,v in self.__dict__.items() if type(v)==list}
        self.non_compare_parameters={k:v for k, v in self.__dict__.items() if k not in self.compare_parameters and k !='compare_parameters'}
        self.compare_parameters_names= list(self.compare_parameters.keys())
        self.scenarios_list=list(itertools.product(*list(self.compare_parameters.values())))
        self.num_scenarios=len(self.scenarios_list)
        self.scenarios_df=pd.DataFrame(self.scenarios_list, columns =self.compare_parameters_names)

        self.classifiers=[]

    def grid(self):
        return self.scenarios_df



class Feature_Extraction():

    def __init__(self, DEFAULT_SETTINGS=FEATURE_EXTRACTION_PIPELINE_SETTINGS, **kwargs):

        for k, v in kwargs.items():
            setattr(self, k, v)
        for k, v in DEFAULT_SETTINGS.items():
            if k not in kwargs.keys():
                setattr(self, k, v)

        if 'device' not in kwargs.keys(): self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_processor=Data_Processor(**self.__dict__)
        self.feature_extractor=Feature_Extractor(dataloader=self.data_processor.dataloader, **self.__dict__)

    def info(self):
        info=pd.DataFrame.from_dict(({key:str(value) for key, value in self.__dict__.items()}).items())
        info.columns=['Property', 'Value']
        return info

    def run(self, **kw):
        if 'feature_table' in kw.keys():
            print ('Loading Extracted Features')
            self.feature_table=kw['feature_table']
            self.feature_names=kw['feature_names']
        elif 'feature_table' not in self.__dict__.keys():
            print ('Running Feature Extraction.')
            self.feature_extractor.run()
            self.feature_table=self.feature_extractor.feature_table
            self.feature_names=self.feature_extractor.feature_names
        return self.feature_table

    def export(self, output_path):
        try:
            outfile=open(output_path,'wb')
            pickle.dump(self,outfile)
            outfile.close()
            print ('Pipeline exported successfully.')
        except:
            raise TypeError('Error! Pipeline could not be exported.')
