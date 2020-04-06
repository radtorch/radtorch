# Copyright (C) 2020 RADTorch and Mohamed Elbanan, MD
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see https://www.gnu.org/licenses/

from radtorch.settings import *
from radtorch.model import *
from radtorch.data import *
from radtorch.vis import *
from radtorch.general import *
from radtorch.dataset import *



def load_pipeline(target_path):
    infile=open(target_path,'rb')
    pipeline=pickle.load(infile)
    infile.close()
    return pipeline


class Pipeline():

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        for k, v  in self.DEFAULT_SETTINGS.items():
            if k not in kwargs.keys():
                setattr(self, k, v)

        if 'device' not in kwargs.keys():
            self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data_subsets=['dataset', 'train_dataset', 'valid_dataset', 'test_dataset']

        # Create Initial Master Dataset
        if isinstance(self.table, pd.DataFrame): self.dataset=Dataset_from_table(**kwargs)
        else: self.dataset=Dataset_from_folder(**kwargs)
        self.num_output_classes=len(self.dataset.classes)
        self.dataloader=torch.utils.data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        # Custom Resize Adjustement
        if isinstance(self.custom_resize, bool): self.resize=model_dict[self.model_arch]['input_size']
        elif isinstance(self.custom_resize, int): self.resize=self.custom_resize

        # Create transformations
        if self.is_dicom:
            self.transformations=transforms.Compose([
                    transforms.Resize((self.resize, self.resize)),
                    transforms.transforms.Grayscale(3),
                    transforms.ToTensor()])
        else:
            self.transformations=transforms.Compose([
                transforms.Resize((self.resize, self.resize)),
                transforms.ToTensor()])

        # Calculate Normalization if required
        if self.normalize=='auto':
            mean, std=self.dataset.mean_std()
            self.transformations.transforms.append(transforms.Normalize(mean=mean, std=std))
        elif isinstance (self.normalize, tuple):
            mean, std=self.normalize
            self.transformations.transforms.append(transforms.Normalize(mean=mean, std=std))

        # Recreate Transformed Master Dataset
        if isinstance(self.table, pd.DataFrame): self.dataset=Dataset_from_table(**kwargs, transformations=self.transformations)
        else: self.dataset=Dataset_from_folder(**kwargs, transformations=self.transformations)
        self.dataloader=torch.utils.data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def info(self):
        info=pd.DataFrame.from_dict(({key:str(value) for key, value in self.__dict__.items()}).items())
        info.columns=['Property', 'Value']
        # try:
        for i in self.data_subsets :
            if i in self.__dict__.keys(): info=info.append({'Property':i, 'Value':len(self.__dict__[i])}, ignore_index=True)
            # except:
            #     pass
        return info

    def dataset_info(self, plot=True, fig_size=(500,300)):
        info_dict={}
        for i in self.data_subsets :
            if i in self.__dict__.keys():
                info_dict[i]=show_dataset_info(self.__dict__[i])
                info_dict[i].style.set_caption(i)
        if plot:
            plot_dataset_info(info_dict, plot_size= fig_size)
        else:
            for k, v in info_dict.items():
                display(v)

    def sample(self, figure_size=(10,10), show_labels=True, show_file_name=False):
        show_dataloader_sample(self.dataloader, figure_size=figure_size, show_labels=show_labels, show_file_name=show_file_name)

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

    def classes(self):
        return self.dataset.class_to_idx


class Image_Classification(Pipeline):

    def __init__(self, **kwargs):
        super(Image_Classification, self).__init__(**kwargs, DEFAULT_SETTINGS=IMAGE_CLASSIFICATION_PIPELINE_SETTINGS)
        self.classifiers=[self]

        #Load from predefined datasets or split master dataset
        if self.load_predefined_datatables: self.dataset_dictionary=load_predefined_datatables(data_directory=self.data_directory,is_dicom=self.is_dicom,predefined_datasets=self.load_predefined_datatables,image_path_column=self.image_path_column,image_label_column=self.image_label_column,mode=self.mode,wl=self.wl,transformations=self.transformations )
        else: self.dataset_dictionary=self.dataset.split(valid_percent=self.valid_percent, test_percent=self.test_percent, sample=self.fly)


        # Create train/valid/test datasets and dataloaders
        for k, v in self.dataset_dictionary.items():
            if self.balance_class: setattr(self, k+'_dataset', v.balance())
            else: setattr(self, k+'_dataset', v)
            setattr(self, k+'_dataloader', torch.utils.data.DataLoader(dataset=self.__dict__[k+'_dataset'], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers))


        # Create Training Model
        self.feature_extractor=Feature_Extractor(self.model_arch, self.pre_trained)
        self.train_model=Classifier(feature_extractor=self.feature_extractor, output_classes=self.num_output_classes)

        self.train_model=self.train_model.to(self.device)

        # Create Training Loss Function
        self.loss_function=create_loss_function(self.loss_function)

        # Create Training Optimizer
        self.optimizer=Optimizer(classifier=self.train_model, learning_rate=self.learning_rate)


    def run(self, verbose=True):
        try:
            print ('Starting Image Classification Pipeline Training')
            self.trained_model, self.train_metrics=train_model(
                                                    model=self.train_model,
                                                    train_data_loader=self.train_dataloader,
                                                    valid_data_loader=self.valid_dataloader,
                                                    train_data_set=self.train_dataset,
                                                    valid_data_set=self.valid_dataset,
                                                    loss_criterion=self.loss_function,
                                                    optimizer=self.optimizer,
                                                    epochs=self.train_epochs,
                                                    device=self.device,
                                                    verbose=verbose)
            self.train_metrics=pd.DataFrame(data=self.train_metrics, columns=['Train_Loss', 'Valid_Loss', 'Train_Accuracy', 'Valid_Accuracy'])
        except:
            raise TypeError('Could not train image classification pipeline. Please check provided parameters.')
            pass

    def export_model(self,output_path):
        try:
            torch.save(self.trained_model, output_path)
            print ('Trained classifier exported successfully.')
        except:
            raise TypeError('Error! Trained Model could not be exported.')

    def inference(self, transformations=None, all_predictions=False, *args, **kwargs):
        if transformations==None:
            transformations=self.transformations
        return model_inference( model=self.trained_model,
                                input_image_path=target_image_path,
                                inference_transformations=transformations,
                                all_predictions=all_predictions)

    def confusion_matrix(self, figure_size=(7,7), target_dataset=None, target_classes=None, cmap=None, *args,  **kwargs):
        if target_dataset==None:
            target_dataset=self.test_dataset
        if target_classes==None:
            target_classes=self.dataset.classes

        target_dataset.transformations=self.transformations
        show_nn_confusion_matrix(model=self.trained_model, target_data_set=target_dataset, target_classes=target_classes, figure_size=figure_size, cmap=cmap, device=self.device)

    def roc(self, target_dataset=None, figure_size=(600,400), *args,  **kwargs):
        if target_dataset==None:
            target_dataset=self.test_dataset
        num_classes=len(target_dataset.classes)
        if num_classes <= 2:
            show_roc([self], fig_size=figure_size)
        else:
            raise TypeError('ROC cannot support more than 2 classes at the current time. This will be addressed in an upcoming update.')
            pass

    def misclassified(self, target_dataset=None, num_images=16, figure_size=(10,10), show_table=False, *args,  **kwargs):
        if target_dataset==None:
            target_dataset=self.test_dataset

        target_dataset.trans=self.transformations

        self.misclassified_instances=show_nn_misclassified(model=self.trained_model, target_data_set=target_dataset, transforms=self.transformations,   is_dicom=self.is_dicom, num_of_images=num_images, device=self.device, figure_size=figure_size)

        if show_table:
            return self.misclassified_instances


class Feature_Extraction(Pipeline):

    def __init__(self, **kwargs):
        super(Feature_Extraction, self).__init__(**kwargs, DEFAULT_SETTINGS=FEATURE_EXTRACTION_PIPELINE_SETTINGS)
        self.classifiers=[self]
        self.model=create_model(model_arch=self.model_arch,output_classes=self.num_output_classes,pre_trained=self.pre_trained,unfreeze_weights=self.unfreeze_weights, mode='feature_extraction')

    def num_features(self):
        return model_dict[self.model_arch]['output_features']

    def run(self, verbose=True):
        self.features=[]
        self.labels_idx=[]
        self.img_path_list=[]

        self.model=self.model.to(self.device)

        for i, (imgs, labels, paths) in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
            self.labels_idx=self.labels_idx+labels.tolist()
            self.img_path_list=self.img_path_list+list(paths)
            with torch.no_grad():
                self.model.eval()
                imgs=imgs.to(self.device)
                output=(self.model(imgs)).tolist()
                self.features=self.features+(output)

        self.feature_names=['f_'+str(i) for i in range(0,(model_dict[self.model_arch]['output_features']))]

        feature_table=pd.DataFrame(list(zip(self.img_path_list, self.labels_idx, self.features)), columns=['img_path','label_idx', 'features'])

        feature_table[self.feature_names]=pd.DataFrame(feature_table.features.values.tolist(), index= feature_table.index)

        feature_table=feature_table.drop(['features'], axis=1)

        print (' Features extracted successfully.')

        self.feature_table=feature_table

        if verbose:
            return self.feature_table

        self.features=self.feature_table[self.feature_names]

    def export_features(self,csv_path):
        try:
            self.feature_table.to_csv(csv_path, index=False)
            print ('Features exported to CSV successfully.')
        except:
            print ('Error! No features found. Please check again or re-run the extracion pipeline.')
            pass

    def plot_extracted_features(self, feature_table=None, feature_names=None, num_features=100, num_images=100,image_path_col='img_path', image_label_col='label_idx'):
        if feature_table==None:
            feature_table=self.feature_table
        if feature_names==None:
            feature_names=self.feature_names
        return plot_features(feature_table, feature_names, num_features, num_images,image_path_col, image_label_col)


class Image_Classifier_Selection(Pipeline):

    def __init__(self, DEFAULT_SETTINGS=COMPARE_CLASSIFIER_PIPELINE_SETTINGS, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        for k, v in DEFAULT_SETTINGS.items():
            if k not in self.__dict__.keys():
                setattr(self, k, v)

        self.compare_parameters={k:v for k,v in self.__dict__.items() if type(v)==list}
        self.non_compare_parameters={k:v for k, v in self.__dict__.items() if k not in self.compare_parameters and k !='compare_parameters'}
        self.compare_parameters_names= list(self.compare_parameters.keys())
        self.scenarios_list=list(itertools.product(*list(self.compare_parameters.values())))
        self.num_scenarios=len(self.scenarios_list)
        self.scenarios_df=pd.DataFrame(self.scenarios_list, columns =self.compare_parameters_names)

        self.classifiers=[]

        for x in self.scenarios_list:
            classifier_settings={self.compare_parameters_names[i]: (list(x))[i] for i in range(len(self.compare_parameters_names))}
            classifier_settings.update(self.non_compare_parameters)
            if self.scenarios_list.index(x)!=0: classifier_settings['load_predefined_datatables']=self.data_subsets
            clf=Image_Classification(**classifier_settings)
            if self.scenarios_list.index(x)==0: self.data_subsets={k:v.input_data for k, v in clf.dataset_dictionary.items()}
            self.classifiers.append(clf)

    def grid(self):
        return self.scenarios_df

    def parameters(self):
        return self.compare_parameters_names

    def dataset_info(self, classifier_index=0, **kwargs):
        self.classifiers[classifier_index].dataset_info(**kwargs)

    def sample(self, classifier_index=0, **kwargs):
        self.dataloader=self.classifiers[classifier_index].dataloader
        super().sample(**kwargs);

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
        self.auc_list=show_roc(self.classifiers, fig_size=fig_size)
        self.best_model_auc=max(self.auc_list)
        self.best_model_index=(self.auc_list.index(self.best_model_auc))
        self.best_classifier=self.classifiers[self.best_model_index]

    def best(self, path=None, export_classifier=False, export_model=False):
        try:
            print ('Best Classifier=Model', self.best_model_index)
            print ('Best Classifier AUC =', self.best_model_auc)
            if export_model:
                export(self.best_classifier.trained_model, path)
                print (' Best Model Exported Successfully')
            if export_classifier:
                export(self.best_classifier, path)
                print (' Best Classifier Pipeline Exported Successfully')
        except:
            raise TypeError('Error! ROC and AUC for classifiers have not been estimated. Please run Compare_Image_Classifier.roc.() first')
