'''
Created on 02.05.2022

@author: Attila-Balazs Kis
'''
import csv
import os
from _operator import truediv

import torch


class HAPTDataset(torch.utils.data.Dataset):
    """class to read the HAPT dataset with a specific structure"""
    def __init__(self, dataDir=None, type="Train", selIndividual=None, seq_length=None):      
        self.labels = []
        self.data = []
        
        print('Loading:', dataDir + type+"/X_t" + type[1:] + ".txt", '- for:', type)
        
        x = open(dataDir + type + "/X_t" + type[1:] + ".txt", 'r')
        y = open(dataDir + type + "/y_t" + type[1:] + ".txt", "r")
        fId = open(dataDir + type + "/subject_id_t" + type[1:] + ".txt", "r")
        
        for row, gt, id in zip(x, y, fId):
            if selIndividual != None and not int(id) in selIndividual: 
                pass
            else:
                row_c = [float(k) for k in row.split(' ')]

                self.data.append(row_c)
                self.labels.append([int(gt.strip())])
        print(' - len:', len(self.data))
        # if seq_length is not None:
        #     x = [] 
        #     y = []
        #     for i in range(round(truediv(len(self.data), seq_length))):
        #         if (len(self.data) - i * seq_length) < seq_length:
        #             pass 
        #         else: 
        #             right = (i + 1) * seq_length
        #             x.append(self.data[i * seq_length:right])
        #             y.append(self.labels[i * seq_length:right])
        #     self.data = x
        #     self.labels = y

        if seq_length is not None:
            x = []
            y = []
            for i in range(0, len(self.data) - seq_length + 1):
                x.append(self.data[i:i+seq_length])
                y.append(self.labels[i:i+seq_length])
            self.data = x
            self.labels = y

    def __getitem__(self, index):
        data = self.data[index]
        lbl = self.labels[index]
        return torch.tensor(data), torch.tensor(lbl)

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    ds = HAPTDataset(
        dataDir='data/HAPT/',
        # selIndividual=list(range(0, 5)),
        seq_length=10
    )
    print(len(ds))
    print('=' * 25)
    _x, _y = next(iter(ds))
    # print('X', _x, 'y', _y)
    print('[batch_size ' + str(_x.shape) + ']', _y.shape)
