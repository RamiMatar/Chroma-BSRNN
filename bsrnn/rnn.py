import torch
import torch.nn as nn
from einops import rearrange

class BandSplitRNNBlock(nn.Module):
    def __init__(self, N, K, T, hidden_size = 256, groups = 8, dropout = 0.0, bidirectional = True):
        super(BandSplitRNNBlock, self).__init__()

        self.group_norm = nn.GroupNorm(groups, N)

        self.seq_blstm = nn.LSTM(N, hidden_size, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.seq_linear = nn.Linear(hidden_size * 2, N)

        self.band_blstm = nn.LSTM(N, hidden_size, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.band_linear = nn.Linear(hidden_size * 2, N)

    def forward(self, Z):
        # Z : (batch_size, N, K, T)
        B, N, K, T = Z.shape

        # ------first we do the sequence level module------

        # Desired shape for sequence level module: (B * K, N, T) for GroupNorm, (B * K, T, N) for LSTM and FC  
        X = rearrange(Z, "b n k t -> (b k) n t")
        # we pass through the group norm | shape unchanged
        X = self.group_norm(X)
        X = rearrange(X, "b n t -> b t n")
       # Z = Z.transpose(0,1)
        # pass through the LSTM | output shape: (B * K, T, hidden * 2)
        X, _ = self.seq_blstm(X)
        # pass through the linear layer | output shape: (B * K, T, N)
        X = self.seq_linear(X)
        X = rearrange(X, "(b k) t n -> b n k t", b = B, k = K)
        # Z is now done with the sequence module, with shape (B * K, T, N)
        Z = X + Z


        # ------second we do the band level module------

        # Desired shape for band level module: (B * T, N, K) for GroupNorm, (B * T, K, N) for LSTM and FC
        X = rearrange(Z, "b n k t -> (b t) n k", b = B, k = K)

        # we pass through the group norm | shape unchanged
        X = self.group_norm(X)
        
        X = rearrange(X, "b n k -> b k n")
        # pass through the LSTM | output shape: (B * K, T, hidden * 2)
        X, _ = self.band_blstm(X)
        # pass through the linear layer | output shape: (B * K, T, N)
        X = self.band_linear(X)
        X = rearrange(X, "(b t) k n -> b n k t", b = B, t = T)
        # Z is now done with the band module, with shape (B * T, K, N)
        Z = X + Z

        return Z


class BandSplitRNN(nn.Module):
    def __init__(self, N, K, T, layers = 6, hidden_size = 512, groups = 8, dropout = 0.0, bidirectional = True):
        super(BandSplitRNN, self).__init__()
        self.rnns = nn.ModuleList([BandSplitRNNBlock(N, K, T, hidden_size, groups, dropout, bidirectional) for i in range(layers)])
        self.num_layers = layers
    def forward(self, Z):
        for i in range(self.num_layers):
            Z = self.rnns[i](Z)
        return Z
