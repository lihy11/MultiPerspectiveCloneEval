import torch
import torch.nn as nn
import torch.nn.functional as thfunc


class LinearModel(nn.Module):
    def __init__(self, vocab_size, out_dim=100, hidden_dim=256):
        super(LinearModel, self).__init__()
        self.in_dim = vocab_size
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.layers = nn.Sequential(
            nn.Linear(in_features=self.in_dim, out_features=self.hidden_dim, bias=True),
            nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim, bias=True),
            nn.Linear(in_features=self.hidden_dim, out_features=self.out_dim, bias=True)
        )
        self.activation = thfunc.leaky_relu

    def forward(self, x1, x2):
        """
        :param x:(N, vocab_size)
        :return: out: (N, out_dim)
        """
        out1 = x1
        out2 = x2
        for i in range(3):
            out1 = self.layers[i](out1)
            out1 = self.activation(out1)
            out2 = self.layers[i](out2)
            out2 = self.activation(out2)
        return out1, out2

class PaceModel(nn.Module):
    def __init__(self, vocab_size, out_dim=100, hidden_dim=256):
        super(PaceModel, self).__init__()
        self.in_dim = vocab_size
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.layers = nn.Sequential(
            nn.Linear(in_features=self.in_dim, out_features=self.hidden_dim, bias=True),
            nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim, bias=True),
            nn.Linear(in_features=self.hidden_dim, out_features=self.out_dim, bias=True)
        )
        self.activation = thfunc.leaky_relu

    def forward(self, x1, x2):
        """
        :param x:(N, vocab_size)
        :return: out: (N, out_dim)
        """
        out1 = x1
        out2 = x2
        for i in range(3):
            out1 = self.layers[i](out1)
            out1 = self.activation(out1)
            out2 = self.layers[i](out2)
            out2 = self.activation(out2)
        return torch.unsqueeze(torch.mean(out1, 0), dim=0), torch.unsqueeze(torch.mean(out2, 0), dim=0)