import torch
from copy import deepcopy
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

from .data import *
from .classifier import *
from .extractor import *
from .utils import *



class ClassifierMetrics(): #OK
    '''
    `ClassifierMetrics` class a set of methods that enables quantiative evaluation of a trained image classifier performance.

    Args:
        classifier (ImageClassifier): trained [`ImageClassifier`](../classifier/#radtorch.classifier.ImageClassifier).
        use_best_model (bool): True to use the model with the lowest validation loss.
        device (str): Device to be used for training. Default: 'auto' which automtically detects GPU presence and uses it for feature extraction. Options: 'auto', 'cuda', 'cpu'.

    '''
    def __init__(self, classifier, use_best=True, device='auto'):
        self.classifier = classifier
        self.device = select_device(device)

        if self.classifier.type == 'torch':
            if use_best:
                self.model = self.classifier.best_model
            else:
                self.model = self.classifier.model
        else:
            self.model = self.classifier.feature_extractors['train'].model


    def _predict_img(self, model, img_path, ds, subset):
        with torch.no_grad():
            img = image_to_tensor(img_path, ds.out_channels, ds.transform[subset], ds.WW, ds.WL).unsqueeze(0).to(self.device)
            output = model(img).cpu().detach().numpy()
            if self.classifier.type == 'torch':
                pred = [(i.tolist()).index(max(i.tolist())) for i in output][0]
            elif self.classifier.type == 'sklearn':
                pred = [(i.tolist()).index(max(i.tolist())) for i in self.classifier.best_model.predict_proba(output)][0]
        return pred

    def get_predictions(self, subset):
        '''
        new dataframe is created : self.pred_table, with important columns : `label_id` and `pred_id`
        '''
        ds = self.classifier.dataset
        self.model = self.model.to(self.device)
        self.model.eval()
        df = deepcopy(ds.tables[subset])
        df['pred_id'] = df.apply( lambda row : self._predict_img(self.model, row[ds.path_col], ds, subset) , axis=1)
        df['label_id'] = df.apply( lambda row : ds.class_to_idx[row[ds.label_col]], axis=1)
        self.pred_table = df
        return df

    def confusion_matrix(self, subset='test', figsize=(8,6), cmap='Blues', percent=False): #https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py
        '''
        Returns confusion matrix using target data.
        Code Adapted from https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py

        Args:
            subset (str): subset of the `dataset` to be used. This can be 'train', 'valid', or 'test'.
            figsize (tuple): size of the displayed figure.
            cmap (string): Name of Matplotlib color map to be used. See [Matplotlib colormaps](https://matplotlib.org/stable/tutorials/colors/colormaps.html)
            percent (bool): True to use percentages instead of real values.

        Returns:

            figure: figure containing confusion matrix


        '''
        df = self.get_predictions(subset)
        true_labels, pred_labels = df.label_id, df.pred_id
        cm = metrics.confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=figsize)
        if percent:
            sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap=cmap, xticklabels=self.classifier.dataset.class_to_idx.keys() ,yticklabels=self.classifier.dataset.class_to_idx.keys(), linewidths=1, linecolor='black')
        else:
            sns.heatmap(cm, annot=True, cmap=cmap, xticklabels=self.classifier.dataset.class_to_idx.keys(),yticklabels=self.classifier.dataset.class_to_idx.keys(),  linewidths=1, linecolor='black')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('Confusion Matrix', fontweight='bold')

    def roc(self, subset= 'test', figure_size=(8,6), plot=True):
        '''
        Displays ROC of the trained classifier and returns ROC-AUC.

        Args:
            subset (str): subset of the `dataset` to be used. This can be 'train', 'valid', or 'test'.
            figsize (tuple): size of the displayed figure.
            plot (bool): True to display ROC.

        Returns:
            float: float of ROC-AUC

        '''

        df = self.get_predictions(subset)
        true_labels, pred_labels = df.label_id, df.pred_id
        fpr, tpr, thresholds = metrics.roc_curve(true_labels, pred_labels)
        auc = metrics.roc_auc_score(true_labels, pred_labels)
        self.auc = auc
        if plot:
            sns.set_style("darkgrid")
            fig = plt.figure(figsize=figure_size)
            plt.plot([0, 0.5, 1.0], [0, 0.5, 1.0], linestyle=':')
            plt.plot(fpr, tpr, linestyle='--', lw=1.1,  label = "ROC AUC = {:0.3f}".format(auc))
            plt.xlabel('False Positive Rate (1-Specificity)')
            plt.ylabel('True Positive Rate (Senstivity)')
            plt.title('Receiver Operating Characteristic Curve',y=-0.2 , fontweight='bold')
            plt.legend()
            plt.show()
        else:
            return auc

    def classification_report(self, subset='test'):
        '''
        Returns text report showing the main classification metrics.

        Args:
            subset (str): subset of the `dataset` to be used. This can be 'train', 'valid', or 'test'.

        '''
        df = self.get_predictions(subset)
        return pd.DataFrame(classification_report(df['label_id'], df['pred_id'], target_names=self.classifier.dataset.classes, output_dict=True))


    # def test(self, subset='test'):
    #     test_loss = 0.0
    #     class_correct = list(0. for i in range(len(self.classifier.dataset.class_to_idx.keys())))
    #     class_total = list(0. for i in range(len(self.classifier.dataset.class_to_idx.keys())))
    #
    #     if self.classifier.type == 'torch':
    #         with torch.no_grad():
    #             self.model.eval()
    #             for imgs, labels, idx in self.classifier.dataset.loaders[subset]:
    #                 imgs, labels = imgs.to(self.device), labels.to(self.device)
    #                 output = self.model(imgs.float())
    #                 loss = self.classifier.criterion(output, labels)
    #                 test_loss += loss.item()*imgs.size(0)
    #                 _, pred = torch.max(output, 1)
    #                 correct = np.squeeze(pred.eq(labels.data.view_as(pred)))
    #                 for i in range(imgs.shape[0]):
    #                     label = labels.data[i]
    #                     class_correct[label] += correct[i].item()
    #                     class_total[label] += 1
    #
    #             test_loss = test_loss/len(self.classifier.dataset.loaders['test'].dataset)
    #             print('Overall Test Loss: {:.6f}\n'.format(test_loss))
    #
    #     elif self.classifier.type == 'sklearn':
    #         df = self.get_predictions(subset)
    #         true_labels, pred_labels = df.label_id.values, df.pred_id.values
    #         true_labels = torch.LongTensor(true_labels)
    #         predictions = torch.FloatTensor(pred_labels)
    #         correct = np.squeeze(predictions.eq(true_labels.data.view_as(predictions)))
    #         for i in range(len(true_labels)):
    #             label = true_labels.data[i]
    #             class_correct[label] += correct[i].item()
    #             class_total[label] += 1
    #
    #     for i in range(len(self.classifier.dataset.class_to_idx)):
    #         c = next((k for k, v in self.classifier.dataset.class_to_idx.items() if v == i), None)
    #         if class_total[i] > 0:
    #             print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
    #                 c , 100 * class_correct[i] / class_total[i],
    #                 np.sum(class_correct[i]), np.sum(class_total[i])))
    #
    #         else:
    #             print('Test Accuracy of %5s: N/A (no examples)' % (c))
    #
    #     print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    #         100. * np.sum(class_correct) / np.sum(class_total),
    #         np.sum(class_correct), np.sum(class_total)))
