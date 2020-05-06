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


class RADTorch_Dataset(Dataset):

    def __init__(self, **kwargs): #defines the default parameters for dataset class.
        for k,v in kwargs.items():
            setattr(self, k, v)

        for k, v in DEFAULT_DATASET_SETTINGS.items():
            if k not in kwargs.keys():
                setattr(self, k, v)

    def __getitem__(self, index): #handles how to get an image of the dataset.
        image_path=self.input_data.iloc[index][self.image_path_column]
        if self.is_dicom:
            image=dicom_to_narray(image_path, self.mode, self.wl)
            image=Image.fromarray(image)
        else:
            image=Image.open(image_path).convert('RGB')

        image=self.transformations(image)

        label=self.input_data.iloc[index][self.image_label_column]
        label_idx=[v for k, v in self.class_to_idx.items() if k == label][0]

        return image, label_idx, image_path

    def __len__(self): #returns number of images in dataset.
        return len(self.dataset_files)

    def info(self): #returns table of dataset information.
        return show_dataset_info(self)

    def classes(self): #returns list of classes in dataset.
        return self.classes

    def class_to_idx(self): #returns mapping of classes to class id (dictionary).
        return self.class_to_idx

    def parameters(self): #returns all the parameter names of the dataset.
        return self.__dict__.keys()

    def split(self, **kwargs): #splits dataset into train/valid/split, takes test_percent and valid_percent.
        return split_dataset(dataset=self, **kwargs)

    def balance(self, **kwargs): #solves class imbalance in dataset through over-sampling of classes with less images.
        # return over_sample(dataset=self, **kwargs)
        return balance_dataset(dataset=self, **kwargs)

    def mean_std(self): #calculates mean and standard deviation of dataset.
        self.mean, self.std= calculate_mean_std(torch.utils.data.DataLoader(dataset=self))
        return tuple(self.mean.tolist()), tuple(self.std.tolist())

    def normalize(self, **kwargs): #retruns a normalized dataset with either mean/std of the dataset or a user specified mean/std
        if 'mean' in kwargs.keys() and 'std' in kwargs.keys():
            mean=kwargs['mean']
            std=kwargs['std']
        else:
            mean, std=self.mean_std()
        normalized_dataset=copy.deepcopy(self)
        normalized_dataset.transformations.transforms.append(transforms.Normalize(mean=mean, std=std))
        return normalized_dataset


class Dataset_from_table(RADTorch_Dataset):
    def __init__(self, **kwargs):
        super(Dataset_from_table, self).__init__(**kwargs)
        if isinstance(self.table, pd.DataFrame):
            self.input_data=self.table
        elif self.table != None:
            self.input_data=pd.read_csv(self.input_source)
        else:
            raise TypeError('Error! No label table was selected. Please check.')
        if self.is_dicom: self.dataset_files=[x for x in (self.input_data[self.image_path_column].tolist()) if x.endswith('.dcm')]
        else: self.dataset_files=[x for x in (self.input_data[self.image_path_column].tolist()) if x.endswith(IMG_EXTENSIONS)]
        if self.multi_label == True:
            self.classes=list(np.unique([item for t in self.input_data[self.image_label_column].to_numpy() for item in t]))
            self.class_to_idx=class_to_idx(self.classes)
            self.multi_label_idx=[]
            for i, row in self.input_data.iterrows():
                t=[]
                for u in self.classes:
                    if u in row[self.image_label_column]:
                        t.append(1)
                    else:
                        t.append(0)
                self.multi_label_idx.append(t)
            self.input_data['MULTI_LABEL_IDX']=self.multi_label_idx
        else:
            self.classes= list(self.input_data[self.image_label_column].unique())
            self.class_to_idx=class_to_idx(self.classes)
        if len(self.dataset_files)==0:
            print ('Error! No data files found in directory:', self.data_directory)

        if len(self.classes)    ==0:
            print ('Error! No classes extracted from directory:', self.data_directory)


class Dataset_from_folder(RADTorch_Dataset):
    def __init__(self, **kwargs):
        super(Dataset_from_folder, self).__init__(**kwargs)
        self.classes, self.class_to_idx=root_to_class(self.data_directory)
        self.all_files=list_of_files(self.data_directory)
        if self.is_dicom: self.dataset_files=[x for x in self.all_files  if x.endswith('.dcm')]
        else: self.dataset_files=[x for x in self.all_files if x.endswith(IMG_EXTENSIONS)]
        self.all_classes=[path_to_class(i) for i in self.dataset_files]
        self.input_data=pd.DataFrame(list(zip(self.dataset_files, self.all_classes)), columns=[self.image_path_column, self.image_label_column])
        if len(self.dataset_files)==0:
            print ('Error! No data files found in directory:', self.data_directory)
        if len(self.classes)==0:
            print ('Error! No classes extracted from directory:', self.data_directory)


class Data_Processor():
    '''
    kwargs: sample, device, table, data_directory, is_dicom, mode, wl, normalize, balance_class, batch_size, num_workers, model_arch , custom_resize,
    '''
    def __init__(self, DEFAULT_SETTINGS=DEFAULT_DATASET_SETTINGS, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        for k, v  in DEFAULT_SETTINGS.items():
            if k not in kwargs.keys():
                setattr(self, k, v)
        if 'device' not in kwargs.keys(): self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create Initial Master Table
        if isinstance(self.table, str):
                if self.table!='':
                    self.table=pd.read_csv(self.table)
        elif isinstance(self.table, pd.DataFrame): self.table=self.table
        else:
            classes, class_to_idx=root_to_class(self.data_directory)
            all_files=list_of_files(self.data_directory)
            if self.is_dicom: dataset_files=[x for x in all_files  if x.endswith('.dcm')]
            else: dataset_files=[x for x in all_files if x.endswith(IMG_EXTENSIONS)]
            all_classes=[path_to_class(i) for i in dataset_files]
            self.table=pd.DataFrame(list(zip(dataset_files, all_classes)), columns=[self.image_path_column, self.image_label_column])


        # Sample from dataset if necessary
        if self.sample:
            self.table=self.table.sample(frac=self.sample, random_state=100)

        # Split into test, valid and train
        self.temp_table, self.test_table=train_test_split(self.table, test_size=self.test_percent, random_state=100, shuffle=True)
        self.train_table, self.valid_table=train_test_split(self.temp_table, test_size=(len(self.table)*self.valid_percent/len(self.temp_table)), random_state=100, shuffle=True)

        # Define Transformations
        # 1- Custom Resize Adjustement
        if self.custom_resize in [False, '', 0, None]: self.resize=model_dict[self.model_arch]['input_size']
        elif isinstance(self.custom_resize, int): self.resize=self.custom_resize

        # 2- Image conversion from DICOM
        if 'transformations' not in self.__dict__.keys():
            if self.is_dicom:
                self.transformations=transforms.Compose([
                        transforms.Resize((self.resize, self.resize)),
                        transforms.transforms.Grayscale(3),
                        transforms.ToTensor()])
            else:
                self.transformations=transforms.Compose([
                    transforms.Resize((self.resize, self.resize)),
                    transforms.ToTensor()])



        # 3- Normalize Training Dataset
        self.train_transformations=copy.deepcopy(self.transformations)
        if 'extra_transformations' in self.__dict__.keys():
            for i in self.extra_transformations:
                self.train_transformations.transforms.insert(1, i)
        if isinstance (self.normalize, tuple):
            mean, std=self.normalize
            # self.transformations.transforms.append(transforms.Normalize(mean=mean, std=std))
            self.train_transformations.transforms.append(transforms.Normalize(mean=mean, std=std))

        elif self.normalize!=False:
            log('Error! Selected mean and standard deviation are not allowed.')
            pass

        self.dataset_kwargs=copy.deepcopy(self.__dict__)
        del self.dataset_kwargs['table']

        self.train_dataset_kwargs=copy.deepcopy(self.dataset_kwargs)
        # del self.train_dataset_kwargs['transformations']
        self.train_dataset_kwargs['transformations']=self.train_transformations

        self.master_dataset=Dataset_from_table(table=self.table, **self.dataset_kwargs)
        self.master_dataloader=torch.utils.data.DataLoader(dataset=self.master_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.num_output_classes=len(self.master_dataset.classes)


        if self.type=='nn_classifier':
            self.train_dataset=Dataset_from_table(table=self.train_table, **self.train_dataset_kwargs)
            if self.balance_class:
                self.train_dataset=self.train_dataset.balance(label_col=self.image_label_column, upsample=True)
            self.valid_dataset=Dataset_from_table(table=self.valid_table, **self.dataset_kwargs)
            self.train_dataloader=torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            self.valid_dataloader=torch.utils.data.DataLoader(dataset=self.valid_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        else:
            self.train_dataset=Dataset_from_table(table=self.temp_table,  **self.train_dataset_kwargs)
            if self.balance_class:
                self.train_dataset=self.train_dataset.balance(label_col=self.image_label_column, upsample=True)
            self.train_dataloader=torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        self.test_dataset=Dataset_from_table(table=self.test_table, **self.dataset_kwargs)
        self.test_dataloader=torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def classes(self):
        return self.master_dataset.class_to_idx

    def info(self):
        info=pd.DataFrame.from_dict(({key:str(value) for key, value in self.__dict__.items()}).items())
        info.columns=['Property', 'Value']
        info=info.append({'Property':'master_dataset_size', 'Value':len(self.master_dataset)}, ignore_index=True)
        for i in ['train_dataset', 'valid_dataset','test_dataset']:
            if i in self.__dict__.keys():
                info.append({'Property':i+'_size', 'Value':len(self.__dict__[i])}, ignore_index=True)
        return info

    def dataset_info(self, plot=True, figure_size=(500,300)):
        info_dict={}
        info_dict['dataset']=show_dataset_info(self.master_dataset)
        info_dict['dataset'].style.set_caption('Dataset')
        if 'type' in self.__dict__.keys():
            for i in ['train_dataset','test_dataset']:
                if i in self.__dict__.keys():
                    info_dict[i]= show_dataset_info(self.__dict__[i])
                    info_dict[i].style.set_caption(i)
            if self.type=='nn_classifier':
                if 'valid_dataset' in self.__dict__.keys():
                    info_dict['valid_dataset']= show_dataset_info(self.__dict__['valid_dataset'])
                    info_dict[i].style.set_caption('valid_dataset')

        if plot:
            plot_dataset_info(info_dict, plot_size= figure_size)
        else:
            for k, v in info_dict.items():
                display(v)

    def sample(self, figure_size=(10,10), show_labels=True, show_file_name=False):
        show_dataloader_sample(self.train_dataloader, figure_size=figure_size, show_labels=show_labels, show_file_name=show_file_name)

    def check_leak(self, show_file=False):
        train_file_list=self.train_dataset.input_data[self.image_path_column]
        test_file_list=self.test_dataset.input_data[self.image_path_column]
        leak_files=[]
        for i in train_file_list:
            if i in test_file_list:
                leak_files.append(i)
        log('Data Leak Check: '+str(len(train_file_list))+' train files checked. '+str(len(leak_files))+' common files were found in train and test datasets.')
        if show_file:
            return pd.DataFrame(leak_files, columns='leaked_files')

    def export(self, output_path):
        try:
            outfile=open(output_path,'wb')
            pickle.dump(self,outfile)
            outfile.close()
            log('Data Processor exported successfully.')
        except:
            raise TypeError('Error! Data Processor could not be exported.')


class Feature_Extractor():
    '''
    kwargs: model_arch, pre_trained, unfreeze, device, dataloader
    '''
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self,k,v)
        if self.model_arch not in supported_models:
            log('Error! Provided model architecture is not yet suported. Please use radtorch.settings.supported_models to see full list of supported models.')
            pass
        elif self.model_arch=='vgg11': self.model=torchvision.models.vgg11(pretrained=self.pre_trained)
        elif self.model_arch=='vgg13':  self.model=torchvision.models.vgg13(pretrained=self.pre_trained)
        elif self.model_arch=='vgg16':  self.model=torchvision.models.vgg16(pretrained=self.pre_trained)
        elif self.model_arch=='vgg19':  self.model=torchvision.models.vgg19(pretrained=self.pre_trained)
        elif self.model_arch=='vgg11_bn': self.model=torchvision.models.vgg11_bn(pretrained=self.pre_trained)
        elif self.model_arch=='vgg13_bn': self.model=torchvision.models.vgg13_bn(pretrained=self.pre_trained)
        elif self.model_arch=='vgg16_bn': self.model=torchvision.models.vgg16_bn(pretrained=self.pre_trained)
        elif self.model_arch=='vgg19_bn': self.model=torchvision.models.vgg19_bn(pretrained=self.pre_trained)
        elif self.model_arch=='resnet18': self.model=torchvision.models.resnet18(pretrained=self.pre_trained)
        elif self.model_arch=='resnet34': self.model=torchvision.models.resnet34(pretrained=self.pre_trained)
        elif self.model_arch=='resnet50': self.model=torchvision.models.resnet50(pretrained=self.pre_trained)
        elif self.model_arch=='resnet101': self.model=torchvision.models.resnet101(pretrained=self.pre_trained)
        elif self.model_arch=='resnet152': self.model=torchvision.models.resnet152(pretrained=self.pre_trained)
        elif self.model_arch=='wide_resnet50_2': self.model=torchvision.models.wide_resnet50_2(pretrained=self.pre_trained)
        elif self.model_arch=='wide_resnet101_2': self.model=torchvision.models.wide_resnet101_2(pretrained=self.pre_trained)
        elif self.model_arch=='alexnet': self.model=torchvision.models.alexnet(pretrained=self.pre_trained)

        if 'alexnet' in self.model_arch or 'vgg' in self.model_arch:
            self.model.classifier[6]=torch.nn.Identity()

        #
        # if 'alexnet' in self.model_arch:
        #     self.model.classifier=torch.nn.Linear(in_features=9216, out_features=4096, bias=True)
        # elif 'vgg' in self.model_arch:
        #     self.model.classifier=torch.nn.Linear(in_features=25088, out_features=4096, bias=True)
        elif 'resnet' in self.model_arch:
            self.model.fc=torch.nn.Identity()

        if self.unfreeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def num_features(self):
        return model_dict[self.model_arch]['output_features']

    def run(self, verbose=False):
        if 'balance_class' in self.__dict__.keys() and 'normalize' in self.__dict__.keys():
            log('Running Feature Extraction using '+str(self.model_arch)+' architecture with balance_class = '+str(self.balance_class)+' and normalize = '+str(self.normalize)+".")
        else:
            log('Running Feature Extraction using '+str(self.model_arch)+' architecture')
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
        # self.feature_names=['f_'+str(i) for i in range(0,(model_dict[self.model_arch]['output_features']))]
        self.feature_names=['f_'+str(i) for i in range(0,self.num_features())]
        feature_table=pd.DataFrame(list(zip(self.img_path_list, self.labels_idx, self.features)), columns=['IMAGE_PATH','IMAGE_LABEL', 'FEATURES'])
        feature_table[self.feature_names]=pd.DataFrame(feature_table.FEATURES.values.tolist(), index= feature_table.index)
        feature_table=feature_table.drop(['FEATURES'], axis=1)
        log('Features extracted successfully.')
        self.feature_table=feature_table
        self.features=self.feature_table[self.feature_names]
        if verbose:
            print (self.feature_table)

    def export_features(self,csv_path):
        try:
            self.feature_table.to_csv(csv_path, index=False)
            log('Features exported to CSV successfully.')
        except:
            log('Error! No features found. Please check again or re-run the extracion pipeline.')
            pass

    def plot_extracted_features(self, feature_table=None, feature_names=None, num_features=100, num_images=100,image_path_col='IMAGE_PATH', image_label_col='IMAGE_LABEL'):
        if feature_table==None:
            feature_table=self.feature_table
        if feature_names==None:
            feature_names=self.feature_names
        return plot_features(feature_table, feature_names, num_features, num_images,image_path_col, image_label_col)

    def export(self, output_path):
        try:
            outfile=open(output_path,'wb')
            pickle.dump(self,outfile)
            outfile.close()
            log('Feature Extractor exported successfully.')
        except:
            raise TypeError('Error! Feature Extractor could not be exported.')


class Classifier(object):
  '''
  kwargs: feature_table (in case the split is to be done at classifier), extracted_feature_dictionary (dictionary of train/test features), parameters, image_label_column, image_path_column, transformations/model (for prediction)
  '''
  def __init__(self, DEFAULT_SETTINGS=CLASSIFER_DEFAULT_SETTINGS, **kwargs):
    for k,v in kwargs.items():
      setattr(self,k,v)
    for k, v  in DEFAULT_SETTINGS.items():
        if k not in kwargs.keys():
            setattr(self, k, v)

    if 'feature_table' in self.__dict__.keys():
        if isinstance(self.feature_table, str):
            try:
                self.feature_table=pd.read_csv(self.feature_table)
            except:
                log('Loading feature table failed. Please check the location of the feature table.')
                pass


    if 'extracted_feature_dictionary' in self.__dict__.keys():
        # self.feature_names=[x for x in self.feature_table.columns if x not in [self.image_label_col,self.image_path_col]]
        # self.train_features=self.train_features[self.feature_names]
        # self.test_features=self.test_features.iloc[self.feature_names]
        # self.train_labels=self.train_features.iloc[self.fimage_label_col]
        # self.test_labels=self.test_features.iloc[self.fimage_label_col]
        self.feature_names=self.extracted_feature_dictionary['train']['features_names']
        self.train_features=self.extracted_feature_dictionary['train']['features']
        self.train_labels=np.array(self.extracted_feature_dictionary['train']['labels'])
        self.test_features=self.extracted_feature_dictionary['test']['features']
        self.test_labels=np.array(self.extracted_feature_dictionary['test']['labels'])

    else:
        self.feature_names=[x for x in self.feature_table.columns if x not in [self.image_label_column,self.image_path_column]]
        self.labels=self.feature_table[self.image_label_column]
        self.features=self.feature_table[self.feature_names]
        self.train_features,  self.test_features, self.train_labels, self.test_labels=train_test_split(self.features, self.labels, test_size=self.test_percent, random_state=100)

    if self.interaction_terms:
        log('Creating Interaction Terms for Train Dataset.')
        self.train_features=self.create_interaction_terms(self.train_features)
        log('Creating Interaction Terms for Test Dataset.')
        self.test_features=self.create_interaction_terms(self.test_features)
        log('Interaction Terms Created Successfully.')


    self.classifier=self.create_classifier(**self.parameters)
    self.classifier_type=self.classifier.__class__.__name__

  def create_classifier(self, **kw):
    if self.type not in SUPPORTED_CLASSIFIER:
      log('Error! Classifier type not supported. Please check again.')
      pass
    elif self.type=='linear_regression':
      classifier=LinearRegression(n_jobs=-1, **kw)
    elif self.type=='logistic_regression':
      classifier=LogisticRegression(max_iter=10000,n_jobs=-1, **kw)
    elif self.type=='ridge':
      classifier=RidgeClassifier(max_iter=10000, **kw)
    elif self.type=='sgd':
      classifier=SGDClassifier(**kw)
    elif self.type=='knn':
      classifier=KNeighborsClassifier(n_jobs=-1,**kw)
    elif self.type=='decision_trees':
      classifier=tree.DecisionTreeClassifier(**kw)
    elif self.type=='random_forests':
      classifier=RandomForestClassifier(**kw)
    elif self.type=='gradient_boost':
      classifier=GradientBoostingClassifier(**kw)
    elif self.type=='adaboost':
      classifier=AdaBoostClassifier(**kw)
    elif self.type=='xgboost':
      classifier=XGBClassifier(**kw)
    return classifier

  def info(self):
    info=pd.DataFrame.from_dict(({key:str(value) for key, value in self.__dict__.items()}).items())
    info.columns=['Property', 'Value']
    return info

  def run(self):
    self.scores=[]
    self.train_metrics=[]
    if self.cv:
      if self.stratified:
        kf=StratifiedKFold(n_splits=self.num_splits, shuffle=True, random_state=100)
        log('Training '+str(self.classifier_type)+ ' with '+str(self.num_splits)+' split stratified cross validation.')
      else:
        kf=KFold(n_splits=self.num_splits, shuffle=True, random_state=100)
        log('Training '+str(self.classifier_type)+ ' classifier with '+str(self.num_splits)+' splits cross validation.')
      split_id=0
      for train, test in tqdm(kf.split(self.train_features, self.train_labels), total=self.num_splits):
        self.classifier.fit(self.train_features.iloc[train], self.train_labels[train])
        split_score=self.classifier.score(self.train_features.iloc[test], self.train_labels[test])
        self.scores.append(split_score)
        log('Split '+str(split_id)+' Accuracy = ' +str(split_score))
        self.train_metrics.append([[0],[0],[split_score],[0]])
        split_id+=1
    else:
      log('Training '+str(self.type)+' classifier without cross validation.')
      self.classifier.fit(self.train_features, self.train_labels)
      score=self.classifier.score(self.test_features, self.test_labels)
      self.scores.append(score)
      self.train_metrics.append([[0],[0],[score],[0]])
    self.scores = np.asarray(self.scores )
    self.classes=self.classifier.classes_.tolist()
    log(str(self.classifier_type)+ ' model training finished successfully.')
    log(str(self.classifier_type)+ ' overall training accuracy: %0.2f (+/- %0.2f)' % ( self.scores .mean(),  self.scores .std() * 2))
    self.train_metrics = pd.DataFrame(data=self.train_metrics, columns = ['Train_Loss', 'Valid_Loss', 'Train_Accuracy', 'Valid_Accuracy'])
    return self.classifier, self.train_metrics

  def average_cv_accuracy(self):
    if self.cv:
      return self.scores.mean()
    else:
      log('Error! Training was done without cross validation. Please use test_accuracy() instead.')

  def test_accuracy(self) :
    acc= self.classifier.score(self.test_features, self.test_labels)
    return acc

  def confusion_matrix(self,title='Confusion Matrix',cmap=None,normalize=False,figure_size=(8,6)):
    pred_labels=self.classifier.predict(self.test_features)
    true_labels=self.test_labels
    cm = metrics.confusion_matrix(true_labels, pred_labels)
    show_confusion_matrix(cm=cm,
                          target_names=self.classes,
                          title=title,
                          cmap=cmap,
                          normalize=normalize,
                          figure_size=figure_size
                          )

  def roc(self, **kw):
    show_roc([self], **kw)

  def predict(self, input_image_path, all_predictions=False, classifier=None, transformations=None, **kw):
    '''
    Works as a part of pipeline Only
    '''
    if classifier==None:
        classifier=self.classifier

    if transformations==None:
        transformations=self.data_processor.transformations

    model=self.feature_extractor.model

    if input_image_path.endswith('dcm'):
        target_img=dicom_to_pil(input_image_path)
    else:
        target_img=Image.open(input_image_path).convert('RGB')

    target_img_tensor=transformations(target_img)
    target_img_tensor=target_img_tensor.unsqueeze(0)

    with torch.no_grad():
        model.to('cpu')
        target_img_tensor.to('cpu')
        model.eval()
        out=model(target_img_tensor)
    image_features=pd.DataFrame(out, columns=self.feature_names)

    class_to_idx = self.data_processor.classes()

    if all_predictions:
        try:
            A = self.data_processor.classes().keys()
            B = self.data_processor.classes().values()
            C = self.classifier.predict_proba(image_features)[0]
            C = [("%.4f" % x) for x in C]
            return pd.DataFrame(list(zip(A, B, C)), columns=['LABEL', 'LAEBL_IDX', 'PREDICTION_ACCURACY'])
        except:
            log('All predictions could not be generated. Please set all_predictions to False.')
            pass
    else:
        prediction=self.classifier.predict(image_features)

        return (prediction[0], [k for k,v in class_to_idx.items() if v==prediction][0])

  def misclassified(self, num_of_images=4, figure_size=(5,5), table=False, **kw): # NEEDS CHECK FILE PATH !!!!!
      pred_labels=(self.classifier.predict(self.test_features)).tolist()
      true_labels=self.test_labels.tolist()
      accuracy_list=[0.0]*len(true_labels)

      y = copy.deepcopy(self.test_features)
      paths=[]
      for i in y.index.tolist():paths.append(self.test_feature_extractor.feature_table.iloc[i]['IMAGE_PATH'])  # <<<<< this line was changed .. check. / Accuracy not showing correctly !!

      misclassified_dict=misclassified(true_labels_list=true_labels, predicted_labels_list=pred_labels, accuracy_list=accuracy_list, img_path_list=paths)
      show_misclassified(misclassified_dictionary=misclassified_dict, transforms=self.data_processor.transformations, class_to_idx_dict=self.data_processor.classes(), is_dicom = self.is_dicom, num_of_images = num_of_images, figure_size =figure_size)
      misclassified_table = pd.DataFrame(misclassified_dict.values())
      if table:
          return misclassified_table

  # NEEDS TESTING
  def coef(self, figure_size=(50,10), plot=False):#BETA
      coeffs = pd.DataFrame(dict(zip(self.feature_names, self.classifier.coef_.tolist())), index=[0])
      if plot:
          coeffs.T.plot.bar(legend=None, figsize=figure_size);
      else:
          return coeffs

  # NEEDS TESTING
  def create_interaction_terms(self, table):#BETA
        self.interaction_features=table.copy(deep=True)
        int_feature_names = self.interaction_features.columns
        m=len(int_feature_names)
        for i in tqdm(range(m)):
            feature_i_name = int_feature_names[i]
            feature_i_data = self.interaction_features[feature_i_name]
            for j in range(i+1, m):
                feature_j_name = int_feature_names[j]
                feature_j_data = self.interaction_features[feature_j_name]
                feature_i_j_name = feature_i_name+'_x_'+feature_j_name
                self.interaction_features[feature_i_j_name] = feature_i_data*feature_j_data
        return self.interaction_features

  def export(self, output_path):
      try:
          outfile=open(output_path,'wb')
          pickle.dump(self,outfile)
          outfile.close()
          log('Classifier exported successfully.')
      except:
          raise TypeError('Error! Classifier could not be exported.')

  def export_trained_classifier(self, output_path):
      try:
          outfile=open(output_path,'wb')
          pickle.dump(self.classifier,outfile)
          outfile.close()
          log('Trained Classifier exported successfully.')
      except:
          raise TypeError('Error! Trained Classifier could not be exported.')

# NEEDS TESTING
class Feature_Selector(Classifier):

    def feature_feature_correlation(self, cmap='Blues', figure_size=(20,15)):
        corrmat = self.features.corr()
        f, ax = plt.subplots(figsize=figure_size)
        sns.heatmap(corrmat, cmap=cmap, linewidths=.1,ax=ax)

    def feature_label_correlation(self, threshold=0.5):
        corrmat = self.feature_table.corr()
        corr_target = abs(corrmat[self.label_column])
        relevant_features = corr_target[corr_target>threshold]
        df = pd.DataFrame(relevant_features)
        df.columns=['Score']
        df.index.rename('Feature')
        best_features_scores=df.sort_values(by=['Score'], ascending=False)
        best_features_names=df.index.tolist()
        best_features_names.remove(self.label_column)
        best_features_table=self.feature_table[df.index.tolist()]
        return best_features_scores, best_features_names, best_features_table

    def univariate(self, test='chi2', num_features=20):
        if test=='chi2':
          selector = SelectKBest(chi2, k=num_features)
        elif test=='anova':
          selector = SelectKBest(f_classif, k=num_features)
        elif test=='mutual_info':
          selector = SelectKBest(mutual_info_classif, k=num_features)
        selector.fit(self.train_features, self.train_labels)
        feature_score=selector.scores_.tolist()
        df=pd.DataFrame(list(zip(self.feature_names, feature_score)), columns=['Feature', 'Score'])
        best_features_scores=df.sort_values(by=['Score'], ascending=False)[:num_features]
        best_features_names=best_features_scores.Feature.tolist()
        best_features_table=self.feature_table[best_features_names+[self.label_column]]
        return best_features_scores, best_features_names, best_features_table

    def variance(self, threshold=0, num_features=20):
        selector=VarianceThreshold(threshold=threshold)
        selector.fit(self.train_features, self.train_labels)
        feature_score=selector.variances_.tolist()
        df=pd.DataFrame(list(zip(self.feature_names, feature_score)), columns=['Feature', 'Score'])
        best_features_scores=df.sort_values(by=['Score'], ascending=False)[:num_features]
        best_features_names=best_features_scores.Feature.tolist()
        best_features_table=self.feature_table[best_features_names+[self.label_column]]
        return best_features_scores, best_features_names, best_features_table

    def rfe(self, step=1, rfe_features=None):
        if 'rfe_feature_rank' not in self.__dict__.keys():
          self.selector=RFE(estimator=self.classifier, n_features_to_select=rfe_features, step=step)
          self.selector.fit(self.train_features, self.train_labels)
          self.rfe_feature_rank=self.selector.ranking_
        df= pd.DataFrame(list(zip(self.feature_names, self.rfe_feature_rank.tolist())), columns=['Feature', 'Rank'])
        best_features_names=[x for x,v in list(zip(G.feature_names, G.selector.support_.tolist())) if v==True]
        best_features_scores=df.sort_values(by=['Rank'], ascending=True)
        best_features_table=self.feature_table[best_features_names+[self.label_column]]
        return best_features_scores, best_features_names, best_features_table

    def rfecv(self, step=1, n_jobs=-1, verbose=0):
        self.rfecv_selector=RFECV(estimator=self.classifier, step=step, cv=StratifiedKFold(self.num_splits),scoring='accuracy', n_jobs=-1, verbose=verbose)
        self.rfecv_selector.fit(self.train_features, self.train_labels)
        self.optimal_feature_number=self.rfecv_selector.n_features_
        self.optimal_features_names=[x for x,v in list(zip(self.feature_names, self.rfecv_selector.support_.tolist())) if v==True]
        self.best_features_table=self.feature_table[self.optimal_features_names+[self.label_column]]
        log('Optimal Number of Features = '+ str(self.optimal_feature_number))
        j = range(1, len(self.rfecv_selector.grid_scores_) + 1)
        i = self.rfecv_selector.grid_scores_
        output_notebook()
        p = figure(plot_width=600, plot_height=400)
        p.line(j, i, line_width=2, color='#1A5276')
        p.line([self.optimal_feature_number]*len(i),i,line_width=2, color='#F39C12', line_dash='dashed')
        p.xaxis.axis_line_color = '#D6DBDF'
        p.yaxis.axis_line_color = '#D6DBDF'
        p.xgrid.grid_line_color=None
        p.yaxis.axis_line_width = 2
        p.xaxis.axis_line_width = 2
        p.xaxis.axis_label = 'Number of features selected. Optimal = '+str(self.optimal_feature_number)
        p.yaxis.axis_label = 'Cross validation score (nb of correct classifications)'
        p.xaxis.major_tick_line_color = '#D6DBDF'
        p.yaxis.major_tick_line_color = '#D6DBDF'
        p.xaxis.minor_tick_line_color = '#D6DBDF'
        p.yaxis.minor_tick_line_color = '#D6DBDF'
        p.yaxis.major_tick_line_width = 2
        p.xaxis.major_tick_line_width = 2
        p.yaxis.minor_tick_line_width = 0
        p.xaxis.minor_tick_line_width = 0
        p.xaxis.major_label_text_color = '#99A3A4'
        p.yaxis.major_label_text_color = '#99A3A4'
        p.outline_line_color = None
        p.toolbar.autohide = True
        p.title.text='Recursive Feature Elimination with '+str(self.num_splits)+'-split Cross Validation'
        p.title_location='above'
        show(p)
        return self.optimal_features_names, self.best_features_table

    def tsne(self, feature_table=None, figure_size=(800, 800), colormap=COLORS3, **kwargs):
        if isinstance(feature_table, pd.DataFrame):
            y = feature_table
        else:
            y = self.feature_table[self.feature_names+[self.label_column]]
        tsne = TSNE(n_components=2, random_state=0)
        X_2d = tsne.fit_transform(y)
        output_notebook()
        p = figure(tools=TOOLS, plot_width=figure_size[0], plot_height=figure_size[1])
        for i in y.label_idx.unique().tolist():
            p.scatter(X_2d[y[self.label_column] == i, 0], X_2d[y[self.label_column] == i, 1], radius=0.4, fill_alpha=0.6,line_color=None, fill_color=colormap[i])
        p.xaxis.axis_line_color = '#D6DBDF'
        p.yaxis.axis_line_color = '#D6DBDF'
        p.xgrid.grid_line_color=None
        p.ygrid.grid_line_color=None
        p.yaxis.axis_line_width = 2
        p.xaxis.axis_line_width = 2
        p.xaxis.major_tick_line_color = '#D6DBDF'
        p.yaxis.major_tick_line_color = '#D6DBDF'
        p.xaxis.minor_tick_line_color = '#D6DBDF'
        p.yaxis.minor_tick_line_color = '#D6DBDF'
        p.yaxis.major_tick_line_width = 2
        p.xaxis.major_tick_line_width = 2
        p.yaxis.minor_tick_line_width = 0
        p.xaxis.minor_tick_line_width = 0
        p.xaxis.major_label_text_color = '#99A3A4'
        p.yaxis.major_label_text_color = '#99A3A4'
        p.outline_line_color = None
        p.toolbar.autohide = True
        p.title.text='t-distributed Stochastic Neighbor Embedding (t-SNE)'
        p.title_location='above'
        show(p)


class NN_Classifier():
    '''
    kwargs: feature_extractor (REQUIRED), data_processor(REQUIRED) , unfreeze, learning_rate, epochs, optimizer, loss_function, lr_schedules, batch_size, device
    '''
    def __init__(self, DEFAULT_SETTINGS=NN_CLASSIFIER_DEFAULT_SETTINGS, **kwargs):
        for k,v in kwargs.items():
            setattr(self,k,v)

        for k, v in DEFAULT_SETTINGS.items():
            if k not in kwargs.keys():
                setattr(self, k, v)
        if 'device' not in kwargs.keys(): self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if 'feature_extractor' not in self.__dict__.keys() or 'data_processor' not in self.__dict__.keys():
            log('Error! No  Data Processor and/or Feature Selector was supplied. Please Check.')
            pass

        # DATA
        self.output_classes=self.data_processor.num_output_classes
        self.train_dataset=self.data_processor.train_dataset
        self.train_dataloader=self.data_processor.train_dataloader
        self.valid_dataset=self.data_processor.valid_dataset
        self.valid_dataloader=self.data_processor.valid_dataloader
        if self.test_percent>0:
            self.test_dataset=self.data_processor.test_dataset
            self.test_dataloader=self.data_processor.test_dataloader
        self.transformations=self.data_processor.transformations


        # MODEL
        self.model=copy.deepcopy(self.feature_extractor.model)
        self.model_arch=self.feature_extractor.model_arch
        self.in_features=model_dict[self.model_arch]['output_features']

        if self.custom_nn_classifier:
            if 'vgg' in self.model_arch or 'alexnet' in self.model_arch: self.model.classifier=self.custom_nn_classifier
            elif 'resnet' in self.model_arch: self.model.fc=self.custom_nn_classifier

        # elif self.output_features:
        #     if 'vgg' in self.model_arch or 'alexnet' in self.model_arch: self.model.classifier[6]=torch.nn.Sequential(
        #                         torch.nn.Linear(in_features=self.in_features, out_features=self.output_features, bias=True),
        #                         torch.nn.Linear(in_features=self.output_features, out_features=self.output_classes, bias=True),
        #                         torch.nn.LogSoftmax(dim=1))
        #     elif 'resnet' in self.model_arch: self.model.fc=torch.nn.Sequential(
        #                         torch.nn.Linear(in_features=self.in_features, out_features=self.output_features, bias=True),
        #                         torch.nn.Linear(in_features=self.output_features, out_features=self.output_classes, bias=True),
        #                         torch.nn.LogSoftmax(dim=1))

        else:
            if 'vgg' in self.model_arch:
                self.model.classifier=torch.nn.Sequential(
                                torch.nn.Dropout(p=0.5),
                                torch.nn.Linear(in_features=25088, out_features=4096, bias=True),
                                torch.nn.ReLU(inplace=True),
                                torch.nn.Dropout(p=0.5),
                                torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
                                torch.nn.ReLU(inplace=True),
                                torch.nn.Linear(in_features=4096, out_features=self.output_classes, bias=True),
                                torch.nn.LogSoftmax(dim=1))

            elif 'alexnet' in self.model_arch:
                self.model.classifier=torch.nn.Sequential(
                                torch.nn.Dropout(p=0.5),
                                torch.nn.Linear(in_features=9216, out_features=4096, bias=True),
                                torch.nn.ReLU(inplace=True),
                                torch.nn.Dropout(p=0.5),
                                torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
                                torch.nn.ReLU(inplace=True),
                                torch.nn.Linear(in_features=4096, out_features=self.output_classes, bias=True),
                                torch.nn.LogSoftmax(dim=1))


            elif 'resnet' in self.model_arch:
                self.model.fc=torch.nn.Sequential(
                                torch.nn.Linear(in_features=self.in_features, out_features=self.output_classes, bias=True),
                                torch.nn.LogSoftmax(dim=1))

        if self.unfreeze: # This will result in unfreezing and retrain all model layers weights again.
            for param in self.model.parameters():
                param.requires_grad = False




        # Optimizer and Loss Function
        self.loss_function=self.nn_loss_function(type=self.loss_function, **self.loss_function_parameters)
        self.optimizer=self.nn_optimizer(type=self.optimizer, model=self.model, learning_rate=self.learning_rate,  **self.optimizer_parameters)

    def info(self):
        info=pd.DataFrame.from_dict(({key:str(value) for key, value in self.__dict__.items()}).items())
        info.columns=['Property', 'Value']
        for i in ['train_dataset', 'valid_dataset','test_dataset']:
            if i in self.__dict__.keys():
                info.append({'Property':i+' size', 'Value':len(self.__dict__[i])}, ignore_index=True)
        return info

    def nn_optimizer(self, type, model, learning_rate, **kw):
        if type not in supported_nn_optimizers:
            log('Error! Optimizer not supported yet. Please check radtorch.settings.supported_nn_optimizers')
            pass
        elif type=='Adam':
            optimizer=torch.optim.Adam(params=model.parameters(),lr=learning_rate, **kw)
        elif type=='AdamW':
            optimizer=torch.optim.AdamW(params=model.parameters(), lr=learning_rate, **kw)
        elif type=='SparseAdam':
            optimizer=torch.optim.SparseAdam(params=model.parameters(), lr=learning_rate, **kw)
        elif type=='Adamax':
            optimizer=torch.optim.Adamax(params=model.parameters(), lr=learning_rate, **kw)
        elif type=='ASGD':
            optimizer=torch.optim.ASGD(params=model.parameters(), lr=learning_rate, **kw)
        elif type=='RMSprop':
            optimizer=torch.optim.RMSprop(params=model.parameters(), lr=learning_rate, **kw)
        elif type=='SGD':
            optimizer=torch.optim.SGD(params=model.parameters(), lr=learning_rate, **kw)
        log('Optimizer selected is '+type)
        return optimizer

    def nn_loss_function(self, type, **kw):
        if type not in supported_nn_loss_functions:
            log('Error! Loss functions not supported yet. Please check radtorch.settings.supported_nn_loss_functions')
            pass
        elif type== 'NLLLoss':
            loss_function=torch.nn.NLLLoss(**kw),
        elif type== 'CrossEntropyLoss':
            loss_function=torch.nn.CrossEntropyLoss(**kw)
        elif type== 'MSELoss':
            loss_function=torch.nn.MSELoss(**kw)
        elif type== 'PoissonNLLLoss':
            loss_function=torch.nn.PoissonNLLLoss(**kw)
        elif type== 'BCELoss':
            loss_function=torch.nn.BCELoss(**kw)
        elif type== 'BCEWithLogitsLoss':
            loss_function=torch.nn.BCEWithLogitsLoss(**kw)
        elif type== 'MultiLabelMarginLoss':
            loss_function=torch.nn.MultiLabelMarginLoss(**kw)
        elif type== 'SoftMarginLoss':
            loss_function=torch.nn.SoftMarginLoss(**kw)
        elif type== 'MultiLabelSoftMarginLoss':
            loss_function=torch.nn.MultiLabelSoftMarginLoss(**kw)
        elif type== 'CosineSimilarity':
            loss_function=torch.nn.CosineSimilarity(**kw)
        log('Loss function selected is '+type)
        return loss_function

    def run(self, **kw):
        #args: model, train_data_loader, valid_data_loader, train_data_set, valid_data_set,loss_criterion, optimizer, epochs, device, verbose, lr_scheduler,
        model=self.model
        train_data_loader=self.train_dataloader
        valid_data_loader=self.valid_dataloader
        train_data_set=self.train_dataset
        valid_data_set=self.valid_dataset
        loss_criterion=self.loss_function
        optimizer=self.optimizer
        epochs=self.epochs
        device=self.device
        if 'lr_scheduler' in self.__dict__.keys(): lr_scheduler=self.lr_scheduler
        else: lr_scheduler=False

        set_random_seed(100)
        start_time=datetime.now()
        training_metrics=[]
        log('Starting training at '+ str(start_time))
        model=model.to(device)
        for epoch in tqdm(range(epochs)):
            epoch_start=time.time()
            # Set to training mode
            model.train()
            # Loss and Accuracy within the epoch
            train_loss=0.0
            train_acc=0.0
            valid_loss=0.0
            valid_acc=0.0
            for i, (inputs, labels, image_paths) in enumerate(train_data_loader):
                # inputs=inputs.float()
                inputs=inputs.to(device)
                labels=labels.to(device)
                # Clean existing gradients
                optimizer.zero_grad()
                # Forward pass - compute outputs on input data using the model
                outputs=model(inputs)
                # Compute loss
                loss=loss_criterion(outputs, labels)
                # Backpropagate the gradients
                loss.backward()
                # Update the parameters
                optimizer.step()
                # Compute the total loss for the batch and add it to train_loss
                train_loss += loss.item() * inputs.size(0)
                # Compute the accuracy
                ret, predictions=torch.max(outputs.data, 1)
                correct_counts=predictions.eq(labels.data.view_as(predictions))
                # Convert correct_counts to float and then compute the mean
                acc=torch.mean(correct_counts.type(torch.FloatTensor))
                # Compute total accuracy in the whole batch and add to train_acc
                train_acc += acc.item() * inputs.size(0)
            # Validation - No gradient tracking needed
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()
                # Validation loop
                for j, (inputs, labels, image_paths) in enumerate(valid_data_loader):
                    inputs=inputs.to(device)
                    labels=labels.to(device)
                    # Forward pass - compute outputs on input data using the model
                    outputs=model(inputs)
                    # Compute loss
                    loss=loss_criterion(outputs, labels)
                    # Compute the total loss for the batch and add it to valid_loss
                    valid_loss += loss.item() * inputs.size(0)
                    # Calculate validation accuracy
                    ret, predictions=torch.max(outputs.data, 1)
                    correct_counts=predictions.eq(labels.data.view_as(predictions))
                    # Convert correct_counts to float and then compute the mean
                    acc=torch.mean(correct_counts.type(torch.FloatTensor))
                    # Compute total accuracy in the whole batch and add to valid_acc
                    valid_acc += acc.item() * inputs.size(0)
            # Find average training loss and training accuracy
            avg_train_loss=train_loss/len(train_data_set)
            avg_train_acc=train_acc/len(train_data_set)
            # Find average validation loss and training accuracy
            avg_valid_loss=valid_loss/len(valid_data_set)
            avg_valid_acc=valid_acc/len(valid_data_set)
            training_metrics.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
            epoch_end=time.time()
            if lr_scheduler:
                lr_scheduler.step(avg_valid_loss)
            log("Epoch : {:03d}/{} : [Training: Loss: {:.4f}, Accuracy: {:.4f}%]  [Validation : Loss : {:.4f}, Accuracy: {:.4f}%] [Time: {:.4f}s]".format(epoch, epochs, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))
        end_time=datetime.now()
        total_training_time=end_time-start_time
        log('Total training time='+ str(total_training_time))
        self.trained_model=model
        self.train_metrics=training_metrics
        self.train_metrics = pd.DataFrame(data=self.train_metrics, columns = ['Train_Loss', 'Valid_Loss', 'Train_Accuracy', 'Valid_Accuracy'])
        return self.trained_model, self.train_metrics

    def confusion_matrix(self, target_dataset=None, figure_size=(8,6), cmap=None):
        if target_dataset==None:
            target_dataset=self.test_dataset
        target_classes=(self.data_processor.classes()).keys()

        show_nn_confusion_matrix(model=self.trained_model, target_data_set=target_dataset, target_classes=target_classes, device=self.device, figure_size=figure_size, cmap=cmap)

    def roc(self, **kw):
      show_roc([self], **kw)

    def metrics(self, **kw):
        show_metrics([self], **kw)

    def predict(self,  input_image_path, model=None, transformations=None, all_predictions=True, **kw): #input_image_path
        if model==None:
            model=self.trained_model
        if transformations==None:
            transformations=self.transformations

        if input_image_path.endswith('dcm'):
            target_img=dicom_to_pil(input_image_path)
        else:
            target_img=Image.open(input_image_path).convert('RGB')

        target_img_tensor=transformations(target_img)
        target_img_tensor=target_img_tensor.unsqueeze(0)

        with torch.no_grad():
            model.to('cpu')
            target_img_tensor.to('cpu')
            model.eval()
            out=model(target_img_tensor)
            softmax=torch.exp(out).cpu()
            prediction_percentages=softmax.cpu().numpy()[0]
            # prediction_percentages=[i*100 for i in prediction_percentages]
            prediction_percentages = [("%.4f" % x) for x in prediction_percentages]
            _, final_prediction=torch.max(out, 1)
            prediction_table=pd.DataFrame(list(zip(self.data_processor.classes().keys(), [*range(0, len(prediction_percentages), 1)], prediction_percentages)), columns=['label','label_idx', 'prediction_accuracy'])

        if all_predictions:
            return prediction_table
        else:
            return final_prediction.item(), prediction_percentages[final_prediction.item()]

    def misclassified(self, num_of_images=4, figure_size=(5,5), table=False, **kw):
        misclassified_table = show_nn_misclassified(model=self.trained_model, target_data_set=self.test_dataset, num_of_images=num_of_images, device=self.device, transforms=self.data_processor.transformations, is_dicom = self.is_dicom, figure_size=figure_size)
        if table:
            return misclassified_table
