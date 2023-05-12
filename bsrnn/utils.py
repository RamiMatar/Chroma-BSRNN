import matplotlib.pyplot as plt
import torch

def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
      axes = [axes]
    for c in range(num_channels):
      axes[c].specgram(waveform[c], Fs=sample_rate)
      if num_channels > 1:
        axes[c].set_ylabel(f'Channel {c+1}')
      if xlim:
        axes[c].set_xlim(xlim)
    figure.suptitle(title)
    plt.show(block=False)


def show_db_spec(y):
    plt.figure(figsize=(10, 5))
    plt.imshow(10 * torch.log10(y.detach().numpy()), origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title('Spectrogram')
    plt.show()