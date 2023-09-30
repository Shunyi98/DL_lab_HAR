'''
Created on 09.05.2022

@author: Attila-Balazs Kis
'''
import re
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
from torchinfo import summary
from torchvision import utils
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import seaborn as sns

from metrics import calc_metrics
from models import *
from config import device, batch_size


# ##########
def log(message: str, file_name: str) -> None:
    """function to log the given message and if there is a file_name given,
    also output the same message into that file

    :param message: string message
    :param file_name: string file name
    :return: None"""
    print(message)
    if file_name:
        f = open(file_name + ".txt", 'a')
        f.write(message + "\n")
        f.close()


# ##########
class Persistence:
    @staticmethod
    def check_dir_make(directory: str) -> None:
        """function to check whether a given directory path exists, and if not,
        create it at the specified place
        
        :param directory: string path to the wanted directory
        :return: None"""
        import os
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def create_log(file_path: str) -> None:
        """function to create a log file in the specified directory 

        :param file_path: the relative path of the log file
        :return None"""
        if file_path:
            assert len(file_path.split('/')) == 2
            Persistence.check_dir_make(file_path.split('/')[0])

            file_path = file_path + ".txt"
            f = open(file_path, 'w+')
            f.close()

    @staticmethod
    def gen_model_name(base_name: str, epoch: int) -> str:
        """function to return a specific model name using a base name and
        given epoch

        :param base_name: the name to be used as the basis of the model
        :param epoch: the current epoch in which the validation accuracy
        is better than the previous maximum
        :return model name string"""
        return base_name + '-best-' + str(epoch) + '.ckpt'

# ##########
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False         
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss)
        elif score < self.best_score:
            self.counter += 1
            print(f' -- ES counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss)
            self.counter = 0

    def save_checkpoint(self, val_loss):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(' -- ES: found a new best loss: {}'.format(val_loss))
        self.val_loss_min = val_loss

class MLRel:
    LABELS = np.array([
        'WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 
        'STANDING', 'LAYING', 'STAND_TO_SIT', 'SIT_TO_STAND', 'SIT_TO_LIE',
        'LIE_TO_SIT', 'STAND_TO_LIE', 'LIE_TO_STAND'
    ])

    @staticmethod
    def visualize_model(model) -> None:
        if 'CRNN' in model.name:
            print('>>> Visualize CNN weights')
            weights = model.conv.weight.data.cpu().clone()
            for c in np.arange(weights.shape[2]):
                plt.imshow(weights[:, :, c])
                plt.show()

        if 'RNN' in model.name:
            print('>>> Visualize RNN hidden state')
            h0 = model.hs[0][0][-1, -1, :].cpu()
            h_1 = model.hs[-1][0][-1, -1, :].cpu()
            h0 = h0.detach().numpy()
            h_1 = h_1.detach().numpy()
            plt.plot(np.arange(h0.shape[0]), h0, label='first')
            plt.plot(np.arange(h_1.shape[0]), h_1, label='last')
            plt.legend()
            plt.show()
            

    @staticmethod
    def explain_model(model, train, test) -> None:
        """function to explain a model using SHAP(SHapley Additive exPlanations)
        and the train together with test datasets. For faster explanation
        (though losing precision) a subset of values were taken from both
        datasets.
        Show summary plot with average impact of the most influential
        10 features, and bar plot in inferring a random index prediction
        from the test dataset, and which feature affected how the prediction.

        :param model: the PyTorch model object defined in code
        :param train: the train dataset also used for training, but without
        batch structure
        :param test: the test dataset also used for testing but without batch
        structure
        :return: None
        """
        # get the feature names for each sample in the datasets
        warnings.filterwarnings("ignore")
        with open('./data/HAPT/features.txt') as f:
            feats = [l.replace('\n', '').strip() for l in f.readlines()]
        x_train, _ = train[:150]
        x_test, y_test = test[:150]
        
        x_train = torch.tensor(x_train).to(device)
        x_test = torch.tensor(x_test).to(device)
        print('\n> Model under assessment:', model.__str__())

        e = shap.DeepExplainer(model, x_train)
        model.eval()

        shap.initjs()
        print('> Summary plot:')
        shap.summary_plot(
            e.shap_values(x_test), plot_type="bar", feature_names=feats,
            max_display=10,
            class_names=MLRel.LABELS,
            show=False
        )
        _, h = plt.gcf().get_size_inches()
        plt.gcf().set_size_inches(h*3, h)
        plt.show()

        for i, _ in enumerate(MLRel.LABELS):
            ind = np.where(y_test == torch.tensor([i + 1]))[0][0]
            y_hat = torch.argmax(model(x_test[ind])).cpu()
            y = int(y_test[ind][0]) - 1
            print('Ground truth for y_test[{}] = {}'.format(ind, MLRel.LABELS[y]))
            print('Prediction for y_test[{}] = {}'.format(ind, MLRel.LABELS[y_hat]))
            print('\n> Bar plot:'.format(ind))
            shap.bar_plot(
                e.shap_values(x_test[ind:ind+1])[y_hat][0], feature_names=feats, max_display=10
            )

    @staticmethod
    def plot_training_metric(mtype=None, sg_w=3) -> None:
        """function to plot multiple signals of specific type (losses/accuracy),
        computed throughout training for different networks together with
        their names

        :param mtype: tlossv/vloss/acc string
        :param sg_w: the width of the Savitzky-Golay smoothing filter window
        :return None"""
        poss = ['loss', 'acc']
        assert mtype in poss
        if sg_w:
            assert int(sg_w) >= 0, 'SG window width has to be non-negative'
            assert int(sg_w) % 2 == 1, 'SG window width has to be an odd number'
        
        plt.figure(figsize=(12, 7))
        for m in sorted(Path('./checkpoints').glob("*-metrics.npy")):
            metric = np.load(m)[2 if mtype == 'acc' else 0]
            plt.plot(
                np.arange(len(metric)),
                savgol_filter(metric, 0 if not sg_w else int(sg_w), 1),
                label=str(m).split('-')[0].split('/')[-1]
            )
        plt.legend()
        plt.title(mtype)
        plt.show()

    @staticmethod
    def plot_conf_matrix(confusion: np.ndarray, ticklabels: list) -> None:
        """function to plot a confusion matrix using seaborn, function based on
        https://www.realpythonproject.com/understanding-accuracy-recall-precision-f1-scores-and-confusion-matrices/    

        :param confusion: ndarray containing confusion matrix values
        :param ticklabels: the labels to be added to the confusion matrix
        :return: None"""
        plt.figure(figsize=(12, 7))
        sns.heatmap(confusion, annot=True,
            xticklabels=ticklabels, yticklabels=ticklabels
        )
        plt.ylabel("Label")
        plt.xlabel("Predicted")
        plt.show()

    @staticmethod
    def compare_models(models, show_arch: bool = False, conf_cut=None) -> None:
        """function to execute the same set of functionalities for each model,
        such as adding the network architecture, print testing metrics and
        plot a comprehensive visual correlation matrix, for the models to be
        appropriately compares
        
        :param models: list of model paths
        :param show_arch: optional parameter, if True, detailed architecture is
        presented about each network
        :param conf_cut: the amount of cuts made so that the confusion matrix
        is not too crowded with large values
        :return: None"""
        cut = int(conf_cut)
        for i, m in enumerate(models):
            # load and print model summary
            print('\033[92m### {} ###\033[00m'.format(m))
            model = torch.load('./checkpoints/{}' \
                .format(m.split('/')[-1]) + '.ckpt')
            if 'RNN' in model.name:
                print('  > Sequence Length:', model.seq_length)
            if show_arch:
                i_s = (batch_size, 561) if 'RNN' not in model.name \
                                else (batch_size, model.seq_length, 561)
                summ = summary(model, input_size=i_s, verbose=0).__str__()
                print('    ' + summ.replace('\n', '\n    '))

            # ###
            gt = np.load(m + '-gt.npy')
            pred = np.load(m + '-pred.npy')
            a, r, p, f1, auc, conf = calc_metrics(
                labels=gt,
                predictions=pred,
                cut=cut
            )
            print('\n  > Metrics:')
            # "\n   - Rec: ", r, "\n   - Pre: ", p,
            print("   - ACC: ", a, "\n   - F1: ", f1, "\n   - AUC: ", auc)
            gt = gt[:cut]
            cpl_gt = np.bincount(gt.astype(np.int64).flatten())
            nz_cpl_gt = np.where(cpl_gt != 0)[0]  # cpl_gt is 1D

            print('\n  > Count/label:', cpl_gt[1:], 'length:', len(gt))
            ticklabels = list(MLRel.LABELS[nz_cpl_gt - 1])
            if len(conf[0]) > len(nz_cpl_gt):
                ticklabels.insert(0, 0)
            MLRel.plot_conf_matrix(
                conf,
                ticklabels=ticklabels
            )
