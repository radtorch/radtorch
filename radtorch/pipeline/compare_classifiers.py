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

from ..settings import *
from .core import *
from .utils import *
from .image_classification import *


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
        self.scenarios_list=[]
        keys, values = zip(*self.compare_parameters.items()) #http://stephantul.github.io/python/2019/07/20/product-dict/
        for bundle in itertools.product(*values):
            d = dict(zip(keys, bundle))
            d.update(self.non_compare_parameters)
            self.scenarios_list.append(d)
        self.num_scenarios=len(self.scenarios_list)
        self.scenarios_list.sort(key = lambda x: x['type'], reverse=True)
        self.scenarios_df=pd.DataFrame(self.scenarios_list)
        self.classifiers=[]

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
            classifier=Image_Classification(**x)
            log('Starting Training Classifier Number '+str(self.scenarios_list.index(x)))
            classifier.run()
            self.classifiers.append(classifier)
            self.trained_models.append(classifier.trained_model)
            self.master_metrics.append(classifier.train_metrics)
            torch.cuda.empty_cache()
            print('')

    def roc(self, **kw):
        self.auc_list=show_roc([i.classifier for i in self.classifiers], **kw)
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

    def metrics(self, **kw):
        show_metrics([i.classifier for i in self.classifiers], **kw)

    def export(self, output_path):
        try:
            outfile=open(output_path,'wb')
            pickle.dump(self,outfile)
            outfile.close()
            log ('Pipeline exported successfully to '+output_path)
        except:
            log ('Error! Pipeline could not be exported.')
            pass
