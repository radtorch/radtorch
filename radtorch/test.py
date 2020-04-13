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
from radtorch.dicom import *
from radtorch.dataset import *



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
        print (' Features extracted successfully.')
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

  def train(self):
    self.scores=[]
    self.training_metrics=[]
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
        training_metrics.append([_,_,split_score,_])
    else:
      print ('Training', self.type, 'classifier without cross validation.')
      self.classifier.fit(self.train_features, self.train_labels)
      score=self.classifier.score(self.test_features, self.test_labels)
      self.scores.append(score)
      training_metrics.append([_,_,score,_])

    self.scores = np.asarray(self.scores )
    self.classes=self.classifier.classes_.tolist()
    print (self.classifier_type, 'model training finished successfully.')
    print(self.classifier_type, "overall accuracy: %0.2f (+/- %0.2f)" % ( self.scores .mean(),  self.scores .std() * 2))
    return self.classifier, self.training_metrics

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


class NN_Optimizer():
    def __new__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self,k,v)
        if self.type=='Adam':
            self.optimizer=torch.optim.Adam(self.classifier.parameters(),self.learning_rate)
        if self.type=='AdamW':
            self.optimizer=torch.optim.AdamW(self.classifier.parameters(), self.learning_rate)
        if self.type=='SparseAdam':
            self.optimizer=torch.optim.SparseAdam(self.classifier.parameters(), self.learning_rate)
        if self.type=='Adamax':
            self.optimizer=torch.optim.Adamax(self.classifier.parameters(), self.learning_rate)
        if self.type=='ASGD':
            self.optimizer=torch.optim.ASGD(self.classifier.parameters(), self.learning_rate)
        if self.type=='RMSprop':
            self.optimizer=torch.optim.RMSprop(self.classifier.parameters(), self.learning_rate)
        if self.type=='SGD':
            self.optimizer=torch.optim.SGD(self.classifier.parameters(), self.learning_rate)



        return self.optimizer



def nn_loss_function(type):
    try:
        loss_function=supported_loss[type]
        return loss_function
    except:
        raise TypeError('Error! Provided loss function is not supported yet. For complete list of supported models please type radtorch.modelsutils.supported_list()')
        pass

def nn_train(model, train_data_loader, valid_data_loader, train_data_set, valid_data_set,loss_criterion, optimizer, epochs, device, verbose, lr_scheduler):
    '''
    kwargs = model, train_data_loader, valid_data_loader, train_data_set, valid_data_set,loss_criterion, optimizer, epochs, device,verbose
    .. include:: ./documentation/docs/modelutils.md##train_model
    '''
    set_random_seed(100)
    start_time=datetime.datetime.now()
    training_metrics=[]
    if verbose:
        print ('Starting training at '+ str(start_time))


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

            # print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))


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

                #print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))

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

        if verbose:
            print("Epoch : {:03d}/{} : [Training: Loss: {:.4f}, Accuracy: {:.4f}%]  [Validation : Loss : {:.4f}, Accuracy: {:.4f}%] [Time: {:.4f}s]".format(epoch, epochs, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))

    end_time=datetime.datetime.now()
    total_training_time=end_time-start_time
    if verbose:
        print ('Total training time='+ str(total_training_time))

    return model, training_metrics

def nn_inference(model, input_image_path, all_predictions=False, inference_transformations=transforms.Compose([transforms.ToTensor()])):
    '''
    .. include:: ./documentation/docs/modelutils.md##model_inference
    '''
    if input_image_path.endswith('dcm'):
        target_img=dicom_to_pil(input_image_path)
    else:
        target_img=Image.open(input_image_path).convert('RGB')

    target_img_tensor=inference_transformations(target_img)
    target_img_tensor=target_img_tensor.unsqueeze(0)


    with torch.no_grad():
        model.to('cpu')
        target_img_tensor.to('cpu')

        model.eval()

        out=model(target_img_tensor)
        softmax=torch.exp(out).cpu()
        prediction_percentages=softmax.cpu().numpy()[0]
        prediction_percentages=[i*100 for i in prediction_percentages]
        _, final_prediction=torch.max(out, 1)
        prediction_table=pd.DataFrame(list(zip([*range(0, len(prediction_percentages), 1)], prediction_percentages)), columns=['label_idx', 'prediction_percentage'])

    if all_predictions:
        return prediction_table
    else:
        return final_prediction.item(), prediction_percentages[final_prediction.item()]
