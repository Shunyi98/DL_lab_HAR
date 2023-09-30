'''
Created on 09.05.2022

@author: Attila-Balazs Kis
'''
from typing import Tuple

import numpy as np
from mlxtend.evaluate import ftest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix, roc_curve, auc
from _operator import truediv


def accuracy(labels: np.ndarray, predictions: np.ndarray) -> int:
    """function to compute the accuracy between predictions and ground truth
    labels

    :param labels: gt ndarray of the data on which the model was tested
    :param predictions: ndarray of the model predictions
    :return: accuracy metric"""
    correct = (predictions == labels).sum().item()
    return 100 * truediv(correct, len(labels))


def calc_metrics(labels: np.ndarray, predictions: np.ndarray, cut: int) -> Tuple[float, float, float, float, float, np.ndarray]:
    """function to compute, using the sklearn library, meaningful metrics about
    a model's performance using predictions and ground truth labels
    
    :param labels: gt ndarray of the data on which the model was tested
    :param predictions: ndarray of the model predictions
    :param cut: the size of the subset of the labels/predictions to be transformed
    in confusion matrix to analyze
    :return: four metrics, i.e. accuracy, recall, precision and f1 score, and
    a confusion matrix ndarray"""
    average='weighted'

    accuracy = accuracy_score(labels, predictions)
    recall = recall_score(labels, predictions, average=average, zero_division=0)
    precision = precision_score(labels, predictions, average=average, zero_division=0)
    f1 = f1_score(labels, predictions, average=average)

    auc = calcAUC(labels, predictions)
    
    assert cut > 0
    conf = confusion_matrix(labels[:cut], predictions[:cut])

    return accuracy, recall, precision, f1, auc, conf


def calcAUC(labels, predictions):
    # change our predictions and labels to one-hot encoding
    # label = 5, one-hot = (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0) - labels start from 1
    lbls = np.eye(np.max(labels.astype(int)))[labels.astype(int) - 1]
    preds = np.eye(np.max(predictions.astype(int)))[predictions.astype(int) - 1]
    assert len(preds[0]) == 12

    aucs = np.array([])
    for i in np.arange(12):  # 12 classes
        fpr, tpr, _ = roc_curve(lbls[:, i], preds[:, i])
        aucs = np.append(aucs, auc(fpr, tpr))
    return np.average(aucs)


# to test differences in results of two models 
# return F and p value. if the p-value is smaller than alpha (typically is 0.05) then the results 
# deemed to be statistically differ
# http://rasbt.github.io/mlxtend/user_guide/evaluate/ftest/
def fTest(models,labels):
    f, p_value = ftest(labels, models[0],models[1])
    return f, p_value
