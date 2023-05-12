import torch
import torch.nn as nn
import torchaudio

class Chroma(nn.Module):
    def __init__(self, n_fft, sampling_rate, n_chroma = 12):
        super().__init__()
        self.n_fft = n_fft
        self.sampling_rate = sampling_rate
        self.n_chroma = n_chroma
        freq_bins = torch.linspace(0, sampling_rate / 2, n_fft // 2 + 1)
        mappings = self.semitone_mapping(freq_bins)
        self.semitone_bins = []
        for i in range(self.n_chroma):
            self.semitone_bins.append(torch.argwhere(mappings == torch.Tensor([i])))
    
    def semitone_mapping(self, freq):
        # check (self.n_chroma * torch.log2(freq / 440)) % self.n_chroma instead
        return self.n_chroma * torch.log2(freq / 440) % self.n_chroma

    def forward(self, X):
        chromas = []
        for i in range(12):
            chromas.append(torch.sum(X[:,:, self.semitone_bins[0],:], dim=2))
        chromas = torch.cat(chromas, axis = 2)
        return chromas


class OctaveChroma(nn.Module):
    def __init__(self, sampling_rate = 44100, n_fft = 16384 * 2, hop_length = 512):
        super().__init__()
        self.spec = torchaudio.transforms.Spectrogram(n_fft = n_fft, hop_length = hop_length, win_length = n_fft)
        self.chroma = OctaveAwareChroma(n_fft, sampling_rate)

    def forward(self, x):
        spec = torch.abs(self.spec(x)).transpose(2,3)[:,:,:,1:]
        chroma = self.chroma(spec).transpose(2,3)[:,:,48:]
        return chroma


class OctaveAwareChroma(nn.Module):
    # This module transforms a magnitude stft spectrogram to a log-frequency magnitude spectrogram
    # such that each frequency bin is a musical note and its octave (A4 = 440 Hz, A5 = 880 Hz) and
    # does so by calculating a frequency warping matrix W such that W * X = Y where X is the
    # original spectrogram and Y is the transformed octave aware spectrogram.
    # Due to 
    def __init__(self, n_fft = 8192 * 4, sampling_rate = 44100):
        super().__init__()
        self.n_fft = n_fft
        self.sampling_rate = sampling_rate
        self.freq_bins = torch.linspace(sampling_rate / (2 * self.n_fft), sampling_rate / 2, n_fft // 2) # We ignore the DC component
        self.chroma_bins = self.semitone_mapping(self.freq_bins)
        self.init_W_matrix()
        
    def forward(self, X):
        # X is a spectrogram of shape (batch_size, channels, stft_bins, time_bins)
        # W is a matrix of shape (stft_bins, chroma_bins)
        # Y is a spectrogram of shape (batch_size, channels, chroma_bins, time_bins)
        Y = torch.matmul(X, self.W)
        return Y

    def semitone_mapping(self, freq):
        return 12 * torch.log2(freq / (self.sampling_rate / self.n_fft))

    def init_W_matrix(self):
        num_octaves = 14
        notes_per_octave = 12
        num_unique_chroma_bins = num_octaves * notes_per_octave
        weight_matrix = torch.zeros(self.n_fft // 2, num_unique_chroma_bins)

        for i, freq_bin in enumerate(self.freq_bins):
            # Calculate the chroma bin index and its decimal part
            chroma_idx = self.chroma_bins[i].floor().long()
            chroma_decimal = self.chroma_bins[i] - chroma_idx

            if chroma_idx >= 0 and chroma_idx < num_unique_chroma_bins - 1:
                # If the chroma_idx is within bounds, assign weights to the current chroma_idx and the one to the right
                weight_matrix[i, chroma_idx] = 1 - chroma_decimal
                weight_matrix[i, chroma_idx + 1] = chroma_decimal
            elif chroma_idx == num_unique_chroma_bins - 1:
                # If chroma_idx is the last bin, assign the entire weight to it
                weight_matrix[i, chroma_idx] = 1

        self.W = nn.Parameter(weight_matrix, requires_grad=False)


class Transforms(nn.Module):
    def __init__(self, sr = 44100, n_fft = 2048, hop_length = 512
    , win_length=2048, n_mels = 32):
        super().__init__()
        self.stft = torchaudio.transforms.Spectrogram(n_fft = n_fft, hop_length = hop_length, win_length = win_length, power = None)
        self.mel = torchaudio.transforms.MelScale(sample_rate = sr, n_mels = n_mels, n_stft = n_fft // 2 + 1)
        self.chroma = Chroma(n_fft = n_fft, sampling_rate = sr)

    def forward(self, X):
        stft = self.stft(X)
        power_spectrogram = torch.abs(stft).pow(2)
        real = stft.real
        imag = stft.imag
        stft = torch.stack((real,imag), axis = 4)
        mfccs= self.mel(power_spectrogram)
        chromas = self.chroma(power_spectrogram)
        return stft, chromas, mfccs
