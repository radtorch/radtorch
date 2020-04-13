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
from radtorch.test import *




class Data_Preprocessor(): #device, table, data_directory, is_dicom, normalize, balance_class, batch_size, num_workers, model_arch , custom_resize,

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        # for k, v  in self.DEFAULT_SETTINGS.items():
        #     if k not in kwargs.keys():
        #         setattr(self, k, v)

        if 'device' not in kwargs.keys(): self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create Initial Master Dataset
        if isinstance(self.table, pd.DataFrame): self.dataset=Dataset_from_table(**kwargs)
        else: self.dataset=Dataset_from_folder(**kwargs)

        if self.balance_class:self.dataset=self.dataset.balance()


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

    def classes(self):
        return self.dataset.class_to_idx

    def info(self):
        info=pd.DataFrame.from_dict(({key:str(value) for key, value in self.__dict__.items()}).items())
        info.columns=['Property', 'Value']
        info=info.append({'Property':'Dataset', 'Value':len(self.dataset)}, ignore_index=True)
        return info

    def dataset_info(self, plot=False, fig_size=(500,300)):
        info_dict={}
        info_dict['dataset']=show_dataset_info(self.__dict__['dataset'])
        info_dict['dataset'].style.set_caption('Dataset')
        if plot:
            plot_dataset_info(info_dict, plot_size= fig_size)
        else:
            for k, v in info_dict.items():
                display(v)

    def sample(self, figure_size=(10,10), show_labels=True, show_file_name=False):
        show_dataloader_sample(self.dataloader, figure_size=figure_size, show_labels=show_labels, show_file_name=show_file_name)


class Feature_Extractor(): # model_arch, pre_trained, unfreeze, device, dataloader,

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self,k,v)
        if self.model_arch=='vgg11': self.model=torchvision.models.vgg11(pretrained=self.pre_trained)
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
        print ('Features extracted successfully.')
        self.feature_table=feature_table
        self.features=self.feature_table[self.feature_names]
        if verbose:
            print (self.feature_table)

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
      raise TypeError('Error! Classifier type not supported. Please check again.')
    elif self.type=='linear_regression':
      classifier=LinearRegression(n_jobs=-1, **kw)
    elif self.type=='logistic_regression':
      classifier=LogisticRegression(max_iter=10000,n_jobs=-1, **kw)
    elif self.type=='ridge':
      classifier=RidgeClassifier(max_iter=10000, **kw)
    elif self.type=='elasticnet':
      classifier=ElasticNet(**kw)
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
        print ('Training', self.classifier_type, 'with',self.num_splits,'split stratified cross validation.')
      else:
        kf=KFold(n_splits=self.num_splits, shuffle=True, random_state=100)
        print ('Training', self.classifier_type, 'classifier with',self.num_splits,'splits cross validation.')
      for train, test in tqdm(kf.split(self.train_features, self.train_labels), total=self.num_splits):
        self.classifier.fit(self.train_features.iloc[train], self.train_labels.iloc[train])
        split_score=self.classifier.score(self.train_features.iloc[test], self.train_labels.iloc[test])
        self.scores.append(split_score)
        print ('Split Accuracy =',split_score)
        self.train_metrics.append([0,0,split_score,0])
    else:
      print ('Training', self.type, 'classifier without cross validation.')
      self.classifier.fit(self.train_features, self.train_labels)
      score=self.classifier.score(self.test_features, self.test_labels)
      self.scores.append(score)
      self.train_metrics.append([0,0,score,0])

    self.scores = np.asarray(self.scores )
    self.classes=self.classifier.classes_.tolist()
    print (self.classifier_type, 'model training finished successfully.')
    print(self.classifier_type, "overall accuracy: %0.2f (+/- %0.2f)" % ( self.scores .mean(),  self.scores .std() * 2))
    return self.classifier, self.train_metrics

  def average_cv_accuracy(self):
    if self.cv:
      return self.scores.mean()
    else:
      print ('Error! Training was done without cross validation. Please use test_accuracy() instead.')

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


class Feature_Selection(Classifier):

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
        print ('Optimal Number of Features =', self.optimal_feature_number)
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


class NN_Classifier(object):

    def __new__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self,k,v)
        if 'feature_extractor' not in self.__dist__.keys():
            print ('Error! No Feature Selector Architecture was supplied. Please sepcify which feature extractor you want to use.')
            pass
        else:
            self.classifier_type='Neural Network-FCN with Softmax'
            self.model=self.feature_extractor.model
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
        if self.unfreeze: # This will result in unfreezing and retrain all model layers weight again.
            for param in self.model.parameters():
                param.requires_grad = False
        return self.model

    def info(self):
        info=pd.DataFrame.from_dict(({key:str(value) for key, value in self.__dict__.items()}).items())
        info.columns=['Property', 'Value']
        return info

    def train(self, **kw): #model, train_data_loader, valid_data_loader, train_data_set, valid_data_set, loss_criterion, optimizer, epochs, device, verbose, lr_scheduler
        self.trained_model, self.train_metrics=nn_train(model=self.model, **kw)
        return self.trained_model, self.train_metrics
