import torch
import torch.nn as nn
from transforms import Transforms, OctaveChroma
from modules import BandSplit, MaskEstimation, ChromaSplit, ParallelChromaSplit, AttentionChromaSplit
from rnn import BandSplitRNN


class Model(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        for key, value in hparams.items():
            setattr(self, key, value)
        self.bandwidths = [int(bandwidth) for bandwidth in hparams['bandwidths'].split(',')]
        self.K = len(self.bandwidths)
        self.transforms = Transforms(n_fft = self.n_fft, hop_length = self.hop_length, win_length = self.win_length)
        self.bandsplit = BandSplit(self.bandwidths[:], self.N, self.T)
        self.RNN = BandSplitRNN(self.N, self.K, self.T, layers = self.num_layers, hidden_size = self.blstm_hidden_size)
        self.masks = MaskEstimation(self.bandwidths, self.N, self.T)

    def forward(self, X):
        num_channels = X.shape[1]
        X0, _, _ = self.transforms(X)
        X = self.bandsplit(X0)
        X = self.RNN(X)
        X = self.masks(X, num_channels)
        return X, X0



class ChromaModel(nn.Module):
    def __init__(self, hparams, chroma_split='attention'):
        super().__init__()
        for key, value in hparams.items():
            setattr(self, key, value)
        self.bandwidths = [int(bandwidth) for bandwidth in hparams['bandwidths'].split(',')]
        self.K = len(self.bandwidths)
        self.transforms = Transforms(n_fft = self.n_fft, hop_length = self.hop_length, win_length = self.win_length)
        self.chromagram = OctaveChroma()

        if chroma_split == 'attention':
            self.chromasplit = AttentionChromaSplit()
            print("attention chroma mode")
        elif chroma_split == 'group_fc':
            self.chromasplit = ParallelChromaSplit()
            print("forward layer parallel chroma mode")
        else:
            self.chromasplit = ChromaSplit()
            print("forward layers individual chroma mode")

        self.bandsplit = BandSplit(self.bandwidths[:], self.N, self.T)
        self.band_RNN = BandSplitRNN(self.N, self.K, self.T, layers = self.num_bands_layers, hidden_size = self.blstm_hidden_size)
        self.chroma_RNN = BandSplitRNN(self.N, self.K, self.T, layers = self.num_chroma_layers, hidden_size = self.blstm_hidden_size)
        self.combined_RNN = BandSplitRNN(self.N, self.K, self.T, layers = self.num_combined_layers, hidden_size = self.blstm_hidden_size)
        self.masks = MaskEstimation(self.bandwidths, self.N, self.T)
    
    def forward(self, X):
        num_channels = X.shape[1]
        X0, _, _ = self.transforms(X)
        octave_chroma = self.chromagram(X)
        C = self.chromasplit(octave_chroma)
        X = self.bandsplit(X0)
        X = self.band_RNN(X)
        C = self.chroma_RNN(C)
        X = X+C
        X = self.combined_RNN(X)
        X = self.masks(X, num_channels)
        return X, X0
