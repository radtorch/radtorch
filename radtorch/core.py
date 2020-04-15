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



class Data_Processor(): #device, table, data_directory, is_dicom, normalize, balance_class, batch_size, num_workers, model_arch , custom_resize,

    def __init__(self, DEFAULT_SETTINGS=DEFAULT_DATASET_SETTINGS, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        for k, v  in DEFAULT_SETTINGS.items():
            if k not in kwargs.keys():
                setattr(self, k, v)
        if 'device' not in kwargs.keys(): self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        if self.valid_percent or self.test_percent:
            data_split=self.dataset.split(**kwargs)
            for k,v in data_split.items():
                if self.balance_class:
                    ds=v.balance()
                    setattr(self, k+'_dataset', ds)
                    setattr(self, k+'_dataloader', torch.utils.data.DataLoader(dataset=ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers))
                else:
                    setattr(self, k+'_dataset', v)
                    setattr(self, k+'_dataloader', torch.utils.data.DataLoader(dataset=v, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers))

        if self.balance_class:
            self.dataset=self.dataset.balance()
        self.dataloader=torch.utils.data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def classes(self):
        return self.dataset.class_to_idx

    def info(self):
        info=pd.DataFrame.from_dict(({key:str(value) for key, value in self.__dict__.items()}).items())
        info.columns=['Property', 'Value']
        info=info.append({'Property':'Dataset', 'Value':len(self.dataset)}, ignore_index=True)
        if self.valid_percent or self.test_percent:
            for i in ['train_dataset', 'valid_dataset','test_dataset']:
                if i in self.__dict__.keys():
                    info.append({'Property':i+' size', 'Value':len(self.__dict__[i])}, ignore_index=True)

        return info

    def dataset_info(self, plot=False, figure_size=(500,300)):
        info_dict={}
        info_dict['dataset']=show_dataset_info(self.dataset)
        info_dict['dataset'].style.set_caption('Dataset')
        if 'type' in self.__dict__.keys():
            if self.type=='nn_classifier':
                for i in ['train_dataset', 'valid_dataset','test_dataset']:
                    if i in self.__dict__.keys():
                        info_dict[i]= show_dataset_info(self.__dict__[i])
                        info_dict[i].style.set_caption(i)
        if plot:
            plot_dataset_info(info_dict, plot_size= figure_size)
        else:
            for k, v in info_dict.items():
                display(v)

    def sample(self, figure_size=(10,10), show_labels=True, show_file_name=False):
        show_dataloader_sample(self.dataloader, figure_size=figure_size, show_labels=show_labels, show_file_name=show_file_name)


class Feature_Extractor(): #args: model_arch, pre_trained, unfreeze, device, dataloader

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

        if 'vgg' in self.model_arch or 'alexnet' in self.model_arch: self.model.classifier[6]=torch.nn.Identity()
        elif 'resnet' in self.model_arch: self.model.fc=torch.nn.Identity()

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
        self.feature_names=['f_'+str(i) for i in range(0,(model_dict[self.model_arch]['output_features']))]
        feature_table=pd.DataFrame(list(zip(self.img_path_list, self.labels_idx, self.features)), columns=['img_path','label_idx', 'features'])
        feature_table[self.feature_names]=pd.DataFrame(feature_table.features.values.tolist(), index= feature_table.index)
        feature_table=feature_table.drop(['features'], axis=1)
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

    def plot_extracted_features(self, feature_table=None, feature_names=None, num_features=100, num_images=100,image_path_col='img_path', image_label_col='label_idx'):
        if feature_table==None:
            feature_table=self.feature_table
        if feature_names==None:
            feature_names=self.feature_names
        return plot_features(feature_table, feature_names, num_features, num_images,image_path_col, image_label_col)


class Classifier(object):

  def __init__(self, DEFAULT_SETTINGS=CLASSIFER_DEFAULT_SETTINGS, **kwargs):
    for k,v in kwargs.items():
      setattr(self,k,v)
    for k, v  in DEFAULT_SETTINGS.items():
        if k not in kwargs.keys():
            setattr(self, k, v)
    if isinstance(self.feature_table, str):
        self.feature_table=pd.read_csv(self.feature_table)

    self.features=self.feature_table[self.feature_names]
    self.labels=self.feature_table[self.label_column]
    self.train_features,  self.test_features, self.train_labels, self.test_labels=train_test_split(self.features, self.labels, test_size=self.test_percent, random_state=100)

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
      for train, test in tqdm(kf.split(self.train_features, self.train_labels), total=self.num_splits):
        self.classifier.fit(self.train_features.iloc[train], self.train_labels.iloc[train])
        split_score=self.classifier.score(self.train_features.iloc[test], self.train_labels.iloc[test])
        self.scores.append(split_score)
        log('Split Accuracy = ' +str(split_score))
        self.train_metrics.append([[0],[0],[split_score],[0]])
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

  def predict(self, input_image_path, classifier=None, transformations=None, all_predictions=True, **kw):

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
            return pd.DataFrame(list(zip(A, B, C)), columns=['label', 'label_idx', 'prediction_accuracy'])
        except:
            log('All predictions could not be generated. Please set all_predictions to False.')
            pass
    else:
        prediction=self.classifier.predict(image_features)

        return (prediction[0], [k for k,v in class_to_idx.items() if v==prediction][0])

  def misclassified(self, num_of_images=4, figure_size=(5,5), table=False, **kw):
      pred_labels=(self.classifier.predict(self.test_features)).tolist()
      true_labels=self.test_labels.tolist()
      accuracy_list=[0.0]*len(true_labels)

      y = copy.deepcopy(self.test_features)
      paths=[]
      for i in y.index.tolist():paths.append(self.feature_table.iloc[i]['img_path'])

      misclassified_dict=misclassified(true_labels_list=true_labels, predicted_labels_list=pred_labels, accuracy_list=accuracy_list, img_path_list=paths)
      show_misclassified(misclassified_dictionary=misclassified_dict, transforms=self.data_processor.transformations, class_to_idx_dict=self.data_processor.classes(), is_dicom = self.is_dicom, num_of_images = num_of_images, figure_size =figure_size)
      misclassified_table = pd.DataFrame(misclassified_dict.values())
      if table:
          return misclassified_table


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


class NN_Classifier(): #args: feature_extractor (REQUIRED), data_processor(REQUIRED) , unfreeze, learning_rate, epochs, optimizer, loss_function, lr_schedules, batch_size, device

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
        if self.output_features:
            if 'vgg' in self.model_arch or 'alexnet' in self.model_arch: self.model.classifier[6]=torch.nn.Sequential(
                                torch.nn.Linear(in_features=self.in_features, out_features=self.output_features, bias=True),
                                torch.nn.Linear(in_features=self.output_features, out_features=self.output_classes, bias=True),
                                torch.nn.LogSoftmax(dim=1))
            elif 'resnet' in self.model_arch: self.model.fc=torch.nn.Sequential(
                                torch.nn.Linear(in_features=self.in_features, out_features=self.output_features, bias=True),
                                torch.nn.Linear(in_features=self.output_features, out_features=self.output_classes, bias=True),
                                torch.nn.LogSoftmax(dim=1))
        else:
            if 'vgg' in self.model_arch or 'alexnet' in self.model_arch: self.model.classifier[6]=torch.nn.Sequential(
                                torch.nn.Linear(in_features=self.in_features, out_features=self.output_classes, bias=True),
                                torch.nn.LogSoftmax(dim=1))
            elif 'resnet' in self.model_arch: self.model.fc=torch.nn.Sequential(
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
