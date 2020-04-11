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
from radtorch.vis import *




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
    else:
      print ('Training', self.type, 'classifier without cross validation.')
      self.classifier.fit(self.train_features, self.train_labels)
      score=self.classifier.score(self.test_features, self.test_labels)
      self.classifier.append(score)

    self.scores = np.asarray(self.scores )
    self.classes=self.classifier.classes_.tolist()
    print (self.classifier_type, 'model training finished successfully.')
    print(self.classifier_type, "overall accuracy: %0.2f (+/- %0.2f)" % ( self.scores .mean(),  self.scores .std() * 2))

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




class Feature_selection(Classifier):

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




def find_lr(model, train_dataloader, optimizer, device):
    set_random_seed(100)
    training_losses=[]
    learning_rates=[]
    model=model.to(device)
    for i, (inputs, labels, image_paths) in enumerate(train_dataloader):
        model.train()
        inputs=inputs.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()
        outputs=model(inputs)
        loss=loss_criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss = loss.item() * inputs.size(0)
        training_losses.append(train_loss)
        learning_rates.append(optimizer.learning_rate)


    return training_losses, learning_rates
