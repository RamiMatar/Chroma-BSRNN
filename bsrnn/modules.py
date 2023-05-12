import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class BandSplit(torch.nn.Module):
    # Input shape: torch.Size([2, 2, 1025, 259, 2])
    def __init__(self, bandwidths, N, T):
        super(BandSplit, self).__init__()
        self.N = N
        self.bandwidths = bandwidths
        self.K = len(bandwidths)
        self.norm_layers = torch.nn.ModuleList([torch.nn.LayerNorm((T, self.bandwidths[i] * 2)) for i in range(self.K)]) # check layer norm over single dimension - time / freq
        self.fc_layers = torch.nn.ModuleList([torch.nn.Linear(self.bandwidths[i] * 2, self.N) for i in range(self.K)])

    def forward(self, x):
        # Input shape: batch, num_channel, F, T, 2
        x = x.transpose(2,3) # test with .continguous() and .view() and compare one index
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3] * x.shape[4])
        # shape: new_batch, T, F * 2 (complex)
        subbands = []
        for i in range(self.K):
            start = sum(self.bandwidths[:i])
            subband = x[:, :, 2 * start : 2 * (start + self.bandwidths[i])]
            subband = self.norm_layers[i](subband)
            subband = self.fc_layers[i](subband)
            subbands.append(subband.transpose(1,2))
        # subbands : length K list with equal shape tensors (new_batch, T, N)
        Z = torch.stack(subbands, dim = 2)
        return Z
    

class ChromaSplit(nn.Module):
    def __init__(self, num_octaves = 10, semitones = 12, N = 128, K = 24, T = 517):
        super().__init__()
        self.K = K
        self.fc_layers = nn.ModuleList([nn.Linear(num_octaves * semitones, N) for i in range(K)])

        self.norm_layers = nn.ModuleList([nn.LayerNorm(T, num_octaves * semitones) for i in range(K)])

    def forward(self, x):
        # input shape(B, C, F, T)
        # output shape(B, T, K, N)
        outputs = []
        x = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        for i in range(self.K):
            output = self.norm_layers[i](x).transpose(1,2)
            output = self.fc_layers[i](output)
            outputs.append(output.transpose(1,2))
        C = torch.stack(outputs, dim = 2)
        return C

class AttentionChromaSplit(nn.Module):
    def __init__(self, num_octaves = 10, semitones = 12, N = 128, K = 24, T = 517):
        super().__init__()
        self.K = K
        self.query = nn.Linear(num_octaves * semitones, N * K)
        self.key = nn.Linear(num_octaves * semitones, N * K)
        self.value = nn.Linear(num_octaves * semitones, N * K)
        self.norm = nn.LayerNorm((N, T))

    def forward(self, x):
        x = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3]).transpose(1,2)
        q = self.query(x) # B, T, N*K
        k = self.key(x)
        q = torch.stack(torch.chunk(q, self.K, dim = 2), dim = 2) # B , K , T, N
        k = torch.stack(torch.chunk(k, self.K, dim = 2), dim = 2)
        wei = q @ k.transpose(-1,-2) # B, K, T, T
        v = self.value(x)
        v = torch.stack(torch.chunk(v, self.K, dim = 2), dim = 2)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        out = wei @ v
        out = self.norm(out.permute(0,2,3,1))
        out = out.permute(0,2,1,3)
        return out


class ParallelChromaSplit(nn.Module):
    def __init__(self, num_octaves = 10, semitones = 12, N = 128, K = 24, T = 517):
        super().__init__()
        self.K = K
        self.fc = nn.Linear(num_octaves * semitones, N * K)
        self.norm = nn.GroupNorm(K, N * K)

    def forward(self, x):
        # input shape(B, C, F, T)
        # output shape(B, T, K, N)
        x = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3]) # shape: B, 120, T
        x = self.fc(x.transpose(1,2)) # shape: B, T, N * K
        x = self.norm(x.transpose(1,2)) # shape: B, T, N * K
        x = torch.stack(torch.chunk(x, self.K, dim = 1), dim = 2) # shape: B, K, T, N
        return x


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(MLP, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim * 2),
            nn.GLU()
        )
        
    def forward(self, x):
        x = x.transpose(1,2)
        x = self.MLP(x)
        x = x.transpose(1,2)
        return x
    
    
class MaskEstimation(nn.Module):
    def __init__(self, bandwidths, N, T):
        super(MaskEstimation, self).__init__()
        self.K = len(bandwidths)
        self.bandwidths = bandwidths
        self.norm_layers = torch.nn.ModuleList([torch.nn.LayerNorm(T, N) for bandwidth in self.bandwidths])
        self.MLP_layers = torch.nn.ModuleList([MLP(N, bandwidth * 2, N * 4) for bandwidth in self.bandwidths])

    def forward(self, Q, num_channels):
        # Input: Q, shape: (batch_size, N, K, T)
        B, N, K, T = Q.shape
        Q = Q.permute(2,0,1,3)
        subbands = []
        for i in range(self.K):
            subband = self.norm_layers[i](Q[i])
            subband = self.MLP_layers[i](subband)
            # subband shape is (batch * channel, bandwidth * 2, T) -> (batch * channel, T, bandwidth * 2) -> (batch, channel, T, bandwidth, 2) to represent complex numbers
            subband = rearrange(subband, "(b c) (k z) t -> b c k t z", b = B // num_channels, c = num_channels, z = 2)
            subbands.append(subband)
        # subbands is now a list of length K with the chosen bandwidths
        # reconstruct into spectrogram output
        
        spectrogram = torch.cat(subbands, dim = 2)
        return spectrogram
