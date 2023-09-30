'''
Created on 09.05.2022

@author: Attila-Balazs Kis
'''
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64

if __name__ == '__main__':
    print(device)