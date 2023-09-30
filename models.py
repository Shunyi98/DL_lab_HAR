'''
Created on 09.05.2022

@author: Attila-Balazs Kis
'''
from math import log2
import torch
import torch.nn as nn


class FC(nn.Module):
    def __init__(self, input_nodes=561, hidden_nodes=256, output_nodes=12,
                dropout=0.0, name_suffix=''):   
        super(FC, self).__init__()
        # attribute to assign name for checkpoints
        self.name = type(self).__name__ + name_suffix

        self.inp = nn.Linear(input_nodes, hidden_nodes)
        self.act = nn.Tanh()
        self.d = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_nodes, output_nodes)

    def forward(self, inp):
        h = self.act(self.inp(inp))
        d = self.d(h)
        o = self.out(d)
        return o


class RNN(nn.Module):
    def __init__(self, input_nodes=561, hidden_nodes=256, output_nodes=12,
                dropout=0.0, name_suffix='', seq_length=10):
        super(RNN, self).__init__()
        # attribute to assign name for checkpoints
        self.name = type(self).__name__ + name_suffix
        self.seq_length = seq_length
        self.hs = []

        self.inp = nn.Linear(input_nodes, hidden_nodes)
        self.rnn = nn.LSTM(hidden_nodes, hidden_nodes, 2, batch_first=True)
        self.out = nn.Linear(hidden_nodes, output_nodes)

    def step(self, ip, hidden=None):
        # format: batch, 1, features
        inp = self.inp(ip).unsqueeze(1)
        output, hidden = self.rnn(inp, hidden)
        output = self.out(output.squeeze(1))
        return output, hidden

    def forward(self, inputs, hidden=None):
        #format: batch, seq, features
        outputs = torch.zeros(inputs.shape[0], inputs.shape[1], 12)

        for i in range(self.seq_length):
            outputs[:,i], hidden = self.step(inputs[:,i], hidden)
        self.hs.append(hidden)
        return outputs, hidden

class CRNN(nn.Module):
    def __init__(self, input_nodes=561, hidden_nodes=256, output_nodes=12,
                dropout=0.0, name_suffix='', seq_length=10):
        super(CRNN, self).__init__()
        # attribute to assign name for checkpoints
        self.name = type(self).__name__ + name_suffix
        self.seq_length = seq_length
        self.hs = []
        self.kernel_size = 3

        self.conv = nn.Conv1d(
            in_channels=input_nodes,
            out_channels=hidden_nodes,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2
        )
        self.rnn = nn.LSTM(hidden_nodes, hidden_nodes, 2, batch_first=True)
        self.out = nn.Linear(hidden_nodes, output_nodes)

    def step(self, ip, hidden=None):
        # format: batch, 1, features
        output, hidden = self.rnn(ip.unsqueeze(1), hidden)
        output = self.out(output.squeeze(1))
        return output, hidden

    def forward(self, inputs, hidden=None):
        #format: batch, seq, features
        outputs = torch.zeros(inputs.shape[0], inputs.shape[1], 12)

        inp = inputs.transpose(1, 2)
        inp = self.conv(inp)
        inp = inp.transpose(1, 2)
        for i in range(self.seq_length):
            outputs[:, i], hidden = self.step(inp[:, i], hidden)
        self.hs.append(hidden)
        return outputs, hidden
