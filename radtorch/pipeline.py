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
        set_random_seed(100)
        if self.type!='nn_classifier':
            log('Phase 1: Feature Extraction.')
            if 'feature_table' in self.__dict__.keys():
                log('Loading Extracted Features')
                self.feature_table=self.__dict__['feature_table']
                self.feature_names=self.__dict__['feature_names']
            else:
                self.feature_extractor.run()
                self.feature_table=self.feature_extractor.feature_table
                self.feature_names=self.feature_extractor.feature_names

            log('Phase 2: Classifier Training.')
            log ('Running Classifier Training.')
            self.classifier=Classifier(**self.__dict__)
            self.classifier.run()
            self.trained_model=self.classifier
            self.train_metrics=self.classifier.train_metrics
            self.feature_selector=Feature_Selector(type=self.classifier.type, feature_table=self.feature_extractor.feature_table, feature_names=self.feature_extractor.feature_names)
            log ('Classifier Training completed successfully.')
        else:
            self.classifier=NN_Classifier(**self.__dict__)
            self.trained_model, self.train_metrics=self.classifier.run()
            log ('Classifier Training completed successfully.')

    def metrics(self, figure_size=(500,300)):
        return show_metrics([self.classifier],  fig_size=figure_size)

    def export(self, output_path):
        try:
            outfile=open(output_path,'wb')
            pickle.dump(self,outfile)
            outfile.close()
            log ('Pipeline exported successfully to '+output_path)
        except:
            log ('Error! Pipeline could not be exported.')
            pass


class Compare_Image_Classifiers():

    def __init__(self, DEFAULT_SETTINGS=IMAGE_CLASSIFICATION_PIPELINE_SETTINGS, **kwargs):

        for k, v in kwargs.items():
            setattr(self, k, v)
        for k, v in DEFAULT_SETTINGS.items():
            if k not in kwargs.keys():
                setattr(self, k, v)

        if 'device' not in kwargs.keys(): self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.compare_parameters={k:v for k,v in self.__dict__.items() if type(v)==list}
        self.non_compare_parameters={k:v for k, v in self.__dict__.items() if k not in self.compare_parameters and k !='compare_parameters'}
        self.compare_parameters_names= list(self.compare_parameters.keys())
        # self.scenarios_list=list(itertools.product(*list(self.compare_parameters.values())))
        self.scenarios_list=[]
        keys, values = zip(*self.compare_parameters.items()) #http://stephantul.github.io/python/2019/07/20/product-dict/
        for bundle in itertools.product(*values):
            d = dict(zip(keys, bundle))
            d.update(self.non_compare_parameters)
            self.scenarios_list.append(d)
        self.num_scenarios=len(self.scenarios_list)
        self.scenarios_list.sort(key = lambda x: x['type'], reverse=True)
        # self.scenarios_df=pd.DataFrame(self.scenarios_list, columns =self.compare_parameters_names)
        self.scenarios_df=pd.DataFrame(self.scenarios_list)


        self.classifiers=[]
        self.data_processors=[]
        self.feature_extractors=[]

    def grid(self, full=False):
        if full:
            return self.scenarios_df
        else:
            summary_columns=[]
            df=copy.deepcopy(self.scenarios_df)
            df=df.drop(['parameters', 'table'], axis=1)
            for col in df.columns:
                if len(df[col].unique()) > 1:
                    summary_columns.append(col)
            return self.scenarios_df[summary_columns]

    def run(self):
        log('Starting Image Classification Model Comparison Pipeline.')
        self.master_metrics=[]
        self.trained_models=[]


        for x in self.scenarios_list:
            # settings={self.compare_parameters_names[i]: (list(x))[i] for i in range(len(self.compare_parameters_names))}
            # settings.update(self.non_compare_parameters)
            settings=x
            data_processor=Data_Processor(**settings)
            feature_extractor=Feature_Extractor(dataloader=data_processor.dataloader,**settings)
            if (feature_extractor.model_arch, feature_extractor.balance_class, feature_extractor.normalize) not in [(i.model_arch, i.balance_class,i.normalize )for i in self.feature_extractors]:
                if settings['type']=='nn_classifier':
                    self.feature_extractors.append(feature_extractor)
                    self.data_processors.append(data_processor)
                    log('Skipping Phase 1: Feature Extraction. For NN-Classifiers feature extraction is performed during model training.')
                else:
                    log('Phase 1: Feature Extraction.')
                    feature_extractor.run()
                    self.feature_extractors.append(feature_extractor)
                    self.data_processors.append(data_processor)


        log('Phase 2: Classifier Training.')
        for x in tqdm(self.scenarios_list, total=len(self.scenarios_list)):
            # settings={self.compare_parameters_names[i]: (list(x))[i] for i in range(len(self.compare_parameters_names))}
            # settings.update(self.non_compare_parameters)
            settings=x
            feature_extractor=[i for i in self.feature_extractors if settings['model_arch']==i.model_arch and settings['normalize']==i.normalize and settings['balance_class']==i.balance_class][0]
            if settings['type']!='nn_classifier':
                feature_table=feature_extractor.feature_table
                feature_names=feature_extractor.feature_names
                classifier=Image_Classification(feature_table=feature_table, feature_names=feature_names, feature_extractor=feature_extractor, **settings)
            else:
                classifier=Image_Classification(feature_extractor=feature_extractor, **settings)
            log('Starting Training Classifier Number '+str(self.scenarios_list.index(x)))
            classifier.run()
            self.classifiers.append(classifier)
            self.trained_models.append(classifier.trained_model)
            self.master_metrics.append(classifier.train_metrics)
            torch.cuda.empty_cache()
            print('')

    def roc(self, figure_size=(700,400)):
        self.auc_list=show_roc([i.classifier for i in self.classifiers], fig_size=figure_size)
        self.best_model_auc=max(self.auc_list)
        self.best_model_index=(self.auc_list.index(self.best_model_auc))
        self.best_classifier=self.classifiers[self.best_model_index]

    def best(self, export=False):
        try:
            log('Best Classifier = Model '+str(self.best_model_index))
            log('Best Classifier AUC = '+ str(self.best_model_auc))
            if export:
                self.best_classifier.export(output_path=export)
                log('Best Classifier Pipeline Exported Successfully to '+export)
        except:
            log('Error! ROC and AUC for classifiers have not been estimated. Please run Compare_Image_Classifier.roc.() first')
            pass

    def export(self, output_path):
        try:
            outfile=open(output_path,'wb')
            pickle.dump(self,outfile)
            outfile.close()
            log ('Pipeline exported successfully to '+output_path)
        except:
            log ('Error! Pipeline could not be exported.')
            pass


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
        set_random_seed(100)
        if 'feature_table' in kw.keys():
            log('Loading Extracted Features')
            self.feature_table=kw['feature_table']
            self.feature_names=kw['feature_names']
        elif 'feature_table' not in self.__dict__.keys():
            log('Running Feature Extraction.')
            self.feature_extractor.run()
            self.feature_table=self.feature_extractor.feature_table
            self.feature_names=self.feature_extractor.feature_names
        return self.feature_table

    def export(self, output_path):
        try:
            outfile=open(output_path,'wb')
            pickle.dump(self,outfile)
            outfile.close()
            log('Pipeline exported successfully.')
        except:
            raise TypeError('Error! Pipeline could not be exported.')


def load_pipeline(target_path):
    infile=open(target_path,'rb')
    pipeline=pickle.load(infile)
    infile.close()
    return pipeline
