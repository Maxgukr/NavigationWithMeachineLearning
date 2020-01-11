import torch
import numpy as np
import os
import time
from termcolor import cprint

class LSTMNet(torch.nn.Module):
    def __init__(self):
        super(LSTMNet, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_size=12,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.out = torch.nn.Sequential(
            torch.nn.Linear(64, 3),    
            torch.nn.LeakyReLU()
        )



    def forward(self,x):
        zk_predict,(h_n, c_n) = self.lstm(x)  # zk = [batch_size, seq_len, hidden_size]
        num_layers_dir1, batch_size, hidden_size = h_n.shape
        out = self.out(h_n[-1].squeeze(0))
        return out

class TORCHLSTM(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
 
        self.u_max = None
        self.u_min = None
        self.zk_max = None
        self.zk_min = None
        self.lstm_net = LSTMNet()

    def forward_nets(self, u):
        predict_zk = self.lstm_net(u)
        return predict_zk


    def unnormalize_zk(self, zk_normalized):
        zk = zk_normalized * (self.zk_max - self.zk_min) + self.zk_min
        return zk

    def load(self, args, dataset):
        path_lstmnets = os.path.join(args.path_temp, "lstmnets.p")
        if os.path.isfile(path_lstmnets):
            mondict = torch.load(path_lstmnets)
            self.load_state_dict(mondict)
            cprint("IEKF nets loaded", 'green')
        else:
            cprint("IEKF nets NOT loaded", 'yellow')
        self.get_normalize_factors(dataset)


    def get_normalize_factors(self, dataset):
        self.u_max = dataset.normalize_factors['u_max']
        self.u_min = dataset.normalize_factors['u_min']
        self.zk_max = dataset.normalize_factors['zk_max']
        self.zk_min = dataset.normalize_factors['zk_min']
