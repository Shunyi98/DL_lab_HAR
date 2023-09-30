'''
Created on 09.05.2022

@author: Attila-Balazs Kis
'''
import os
import datetime
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import HAPTDataset
from models import FC, RNN, CRNN
from utils import log, Persistence, EarlyStopping, MLRel
from metrics import accuracy
from config import device, batch_size


def train(model = None, saved_model_path=None, train_loader=None,
          val_loader=None, optimizer=None, log_file=None) -> None:
    """function to perform training & validation sequence in a detailed way
    
    :param ...
    :return: None"""
    log('#' * 50, log_file)
    log('Training', log_file)
    
    total_step = len(train_loader)
    log('# Epochs: {}, steps/epoch: {}'.format(num_epochs, total_step), log_file)
    log('#' * 50, log_file)

    acc = 0
    loss_train = []
    loss_val = []
    acc_val = []
    
    best_acc = -np.inf
    best_model_name = Persistence.gen_model_name(
        base_name=saved_model_path,
        epoch=0
    )

    loss_fcn = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=num_epochs // 15, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        trl, vl = [], []
        print('\n=== Epoch {} ==='.format(epoch + 1))
        for i, (signals, labels) in enumerate(train_loader):
            # print(signals, labels)
            labels = labels.to(device, dtype=torch.int64)
            signals = signals.to(device, dtype=torch.float32)

            # Forward pass
            if 'FC' in model.name:
                outputs = model(signals)
            elif 'RNN' in model.name:
                outputs, _ = model(signals)
                outputs = outputs.to(device, dtype=torch.float32)
            else:
                raise NotImplementedError('Model type not existent!')
            # FC
            if len(outputs.shape) == 2:
                loss = loss_fcn(outputs, labels.flatten() - 1)
            # RNN
            # https://discuss.pytorch.org/t/loss-function-and-lstm-dimension-issues/79291/2
            elif len(outputs.shape) == 3:
                # print(outputs.shape, labels.flatten(1).shape)
                # os._exit(1)
                loss = loss_fcn(outputs.permute(0, 2, 1), labels.flatten(1) - 1)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            trl.append(loss.item())

            if (i + 1) % round(total_step / 5) == 0:
                message = ' > Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, num_epochs, i + 1, total_step, loss.item()
                )
                log(message, log_file)
        loss_train.append(np.average(trl))

        # validation
        model.eval()
        with torch.no_grad():    
            pred, gt = [],[]
            for signalsV, labelsV in val_loader:
                labelsV = labelsV.to(device, dtype=torch.int64)
                signalsV = signalsV.to(device, dtype=torch.float32)

                if 'FC' in model.name:
                    outputsV = model(signalsV)
                elif 'RNN' in model.name:
                    outputsV, _ = model(signalsV)
                    outputsV = outputsV.to(device, dtype=torch.float32)
                else:
                    raise NotImplementedError('Model type not existent!')
                # FC
                if len(outputsV.shape) == 2:
                    loss = loss_fcn(outputs, labels.flatten() - 1)

                    gt.extend(labelsV.to(device).cpu().numpy()[-1])
                    pred.append(torch.argmax(outputsV[-1]).cpu() + 1)
                # RNN
                # https://discuss.pytorch.org/t/loss-function-and-lstm-dimension-issues/79291/2
                elif len(outputsV.shape) == 3:
                    loss = loss_fcn(outputs.permute(0, 2, 1), labels.flatten(1) - 1)

                    gt.append(labelsV.flatten().to(device).cpu().numpy()[-1])
                    pred.append(torch.argmax(outputsV[-1], dim=-1)[-1].cpu() + 1)
                vl.append(loss.item())

            loss_val.append(np.average(vl))
            
            acc = accuracy(
                labels=np.asarray(gt,np.float32), 
                predictions=np.asarray(pred)
            )
            acc_val.append(acc)

            if epoch % 2 == 0:
                message = 'Val Accuracy of the model on the {} step: {} %' \
                    . format(i, acc)
                log(message, log_file)

            early_stopping(np.average(vl))

            if early_stopping.early_stop:
                print("Early stopping")
                break

            # if acc > best_acc: 
            #     torch.save(model, gen_model_name(
            #         base_name=saved_model_path,
            #         epoch=epoch
            #     ))
        model.train()

    torch.save(model, saved_model_path + '.ckpt')

    #save the loss values
    loss_t = np.asarray(loss_train)
    loss_v = np.asarray(loss_val)
    acc = np.asarray(acc_val) / 100
    np.save(saved_model_path + '-metrics.npy', np.array((loss_t, loss_v, acc)))


def test(model=None, saved_model_path=None, test_loader=None, log_file=None,
        log_results=None) -> None:
    """function to perform testing sequence in a detailed way
    
    :param ...
    :return: None"""
    log('#' * 50, log_file)
    log('Testing', log_file)
    log('# Steps per 1 epoch: {}'.format(len(test_loader)), log_file)

    log("Loading model: " + saved_model_path + '.ckpt', log_file)
    model = torch.load(saved_model_path + '.ckpt')
    log('#' * 50, log_file)

    # Test the model
    model.eval()
    hiddens = None
    with torch.no_grad():
        pred, gt = [], []
        for signals, labels in test_loader:
            signals = signals.to(device)
            if 'FC' in model.name:
                outputs = model(signals)
            elif 'RNN' in model.name:
                outputs, _ = model(signals)
            else:
                raise NotImplementedError('Model type not existent!')
            # FC
            if len(outputs.shape) == 2:
                gt.extend(labels.cpu().numpy()[-1])
                pred.append(torch.argmax(outputs[-1]).cpu() + 1)
            # RNN
            elif len(outputs.shape) == 3:
                gt.append(labels.flatten().to(device).cpu().numpy()[-1])
                pred.append(torch.argmax(outputs[-1], dim=1)[-1].cpu() + 1)
        gt = np.asarray(gt, np.float32)
        pred = np.asarray(pred)

        if log_results:
            mp = saved_model_path.split('/')
            mp.insert(-1, 'results')
            Persistence.check_dir_make('/'.join(mp[:-1]))
            saved_model_path = '/'.join(mp)
            np.save(saved_model_path + "-pred.npy", pred)
            np.save(saved_model_path + "-gt.npy", gt)
            log("$ results saved for model " + model.name, log_file)

        message = 'Test Accuracy of the model test samples: {} %' \
            .format(accuracy(
                labels=gt,
                predictions=pred
            ))
        log(message, log_file)

    MLRel.visualize_model(model)


if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='train the model')
    parser.add_argument('--train_losses', action='store_true', help='visualize train|val losses')

    parser.add_argument('--test', action='store_true', help='test (& evaluate) the model')
    # for evaluation
    parser.add_argument('--shap', action='store_true', help='SHapley Additive exPlanations')
    parser.add_argument('--comp', action='store_true', help='compare models to base')
    parser.add_argument('--show_arch', action='store_true', help='show architectures when comparing models')
    parser.add_argument('--conf_cut', help='size of test subset -> conf matrix')
    # for comparing existing models based on training performance
    parser.add_argument('--train_cmp', help='compare progress of existing models based on loss/acc')
    parser.add_argument('--sg_w', help='param for model comparison: the width of the Savitzky-Golay smoothing filter')
    args = parser.parse_args()

    # #########################
    # ### variables ###
    dataDir = 'data/HAPT/'
    output_dir = 'checkpoints'
    Persistence.check_dir_make(output_dir)

    # #########################
    # ### models declaration ###
    seq_length = None
    # #####
    # model = FC(name_suffix='')
    # model = FC(hidden_nodes=64, name_suffix='s')

    # #####
    # Recurrent architectures
    # seq_length = 20  # 5 | 20 | 1
    # model = RNN(seq_length=seq_length,
    #     name_suffix='_LSTM_sl{}'.format(seq_length)
    # )

    seq_length = 20  # 5 | 20
    model = CRNN(seq_length=seq_length,
        name_suffix='_LSTM_sl{}'.format(seq_length)
    )

    # #####
    model.to(device)

    # #########################
    # ML variables
    num_epochs = 150
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # #########################
    # data loaders
    batch_size = batch_size
    
    # datasets used for multiple purposes (train, test, SHAP)
    if not args.train_cmp and not args.train_losses:
        train_ds = HAPTDataset(
            dataDir=dataDir,
            type="Train",
            selIndividual=list(range(27)),
            seq_length=seq_length
        )
        test_ds = HAPTDataset(
            dataDir=dataDir,
            type="Test",
            seq_length=seq_length
        )
    if args.train:
        train_loader = DataLoader(
            dataset=train_ds, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(dataset=HAPTDataset(
            dataDir=dataDir,
            type="Train",
            selIndividual=list(range(27,30)),
            seq_length=seq_length
        ), batch_size=batch_size, shuffle=True)
    if args.test:
        test_loader = DataLoader(
            dataset=test_ds, shuffle=False
        )  # test on big data

    # #########################
    # main
    log_file = 'logs/log_{}-{}'.format(
        model.name,
        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    )
    log_file = None
    Persistence.create_log(log_file)

    if args.train: 
        print('MODEL UNDER ASSESSMENT: {}'.format(model.name))
        train(
            model=model,
            saved_model_path='{}/{}'.format(output_dir, model.name),
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            log_file=log_file
        )
    if args.train_losses:
        print('MODEL UNDER ASSESSMENT: {}'.format(model.name))
        losses = np.load('{}/{}'.format(output_dir, model.name) + '-metrics.npy')

        tloss = losses[0]
        vloss = losses[1]
        _x = np.arange(len(tloss))
        plt.plot(_x, tloss, label='train set loss')
        plt.plot(_x, vloss, label='val set loss')
        plt.legend()
        plt.show()
    if args.test:
        test(
            model=model,
            saved_model_path='{}/{}'.format(output_dir, model.name),
            test_loader=test_loader,
            log_file=None,
            log_results=True
        )
    if args.shap:
        MLRel.explain_model(
            torch.load('{}/{}'.format(output_dir, model.name) + '.ckpt'),
            train=train_ds,
            test=test_ds
        )
    if args.comp:
        models = ['./{}/results/'.format(output_dir) + ckpt.split('.')[0] for 
            ckpt in os.listdir('./' + output_dir) if '.ckpt' in ckpt
        ]
        models.sort()
        MLRel.compare_models(models, args.show_arch, args.conf_cut)
    
    if args.train_cmp:
        if args.train_cmp in ['loss', 'acc']:
            MLRel.plot_training_metric(
                args.train_cmp,
                args.sg_w
            )
        else:
            raise Exception('Train metric for comparison can only be [loss|acc]')
