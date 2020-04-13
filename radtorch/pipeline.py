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
from radtorch.dataset import *
from radtorch.core import *



class Image_Classification():

    def __init__(self, DEFAULT_SETTINGS=IMAGE_CLASSIFICATION_PIPELINE_SETTINGS, **kwargs):

        for k, v in kwargs.items():
            setattr(self, k, v)
        for k, v in DEFAULT_SETTINGS.items():
            if k not in kwargs.keys():
                setattr(self, k, v)

        if 'device' not in kwargs.keys(): self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if 'data_processor' not in self.__dict__.keys(): self.data_processor=Data_Processor(**self.__dict__)
        if 'feature_extractor' not in self.__dict__.keys(): self.feature_extractor=Feature_Extractor(dataloader=self.data_processor.dataloader, **self.__dict__)

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
        self.feature_selector=Feature_Selector(type=self.classifier.type, feature_table=self.feature_extractor.feature_table, feature_names=self.feature_extractor.feature_names)
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


class Compare_Image_Classifiers():

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
        self.data_processors=[]
        self.feature_extractors=[]


        for x in self.scenarios_list:
            settings={self.compare_parameters_names[i]: (list(x))[i] for i in range(len(self.compare_parameters_names))}
            settings.update(self.non_compare_parameters)
            classifier=Image_Classification(**settings)
            self.feature_extractors.append(classifier.feature_extractor)
            classifier.feature_extractor=[i for i in self.feature_extractors if i.model_arch==classifer.model_arch][0]
            self.classifiers.append(classifier)



    def grid(self):
        return self.scenarios_df

    def run(self):
      self.master_metrics=[]
      self.trained_models=[]
      for i in tqdm(self.classifiers, total=len(self.classifiers)):
        print ('Starting Training Classifier Number',self.classifiers.index(i))
        i.run()
        self.trained_models.append(i.trained_model)
        self.master_metrics.append(i.train_metrics)
        torch.cuda.empty_cache()

    def roc(self, fig_size=(700,400)):
        self.auc_list=show_roc([i.classifier for i in self.classifiers], fig_size=fig_size)
        self.best_model_auc=max(self.auc_list)
        self.best_model_index=(self.auc_list.index(self.best_model_auc))
        self.best_classifier=self.classifiers[self.best_model_index]

    def best(self, export=False):
        try:
            print ('Best Classifier=Model', self.best_model_index)
            print ('Best Classifier AUC =', self.best_model_auc)
            if export:
                export(self.best_classifier, export)
                print (' Best Classifier Pipeline Exported Successfully')
        except:
            raise TypeError('Error! ROC and AUC for classifiers have not been estimated. Please run Compare_Image_Classifier.roc.() first')


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
