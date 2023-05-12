import torch
import torchaudio
import numpy as np
import os
import fast_bss_eval
import museval
import tqdm.notebook as tq
import matplotlib.pyplot as plt
import librosa

def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
    if xmax:
      axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)

def zero_pad(signal, segment_samples = 44100 * 6):
    # assumption: even number of samples in a segment
    hop_length = segment_samples // 2
    if signal.shape[1] % hop_length != 0:
        num_zeros = hop_length - (signal.shape[1] % hop_length)
        zero_pad = torch.zeros(signal.shape[0], num_zeros, device = signal.device)
        signal = torch.cat([signal, zero_pad], dim = 1)
    return signal

def split_to_segments(signal, segment_samples = 44100 * 6, overlap = 0.5):
    # input shape: (#channel, samples)
    # output shape: (#channels, #segments, segment_samples)
    start, end = 0, segment_samples
    segments = []
    hop_length = int(segment_samples * (1 - overlap))
    while end <= signal.shape[1]:
        segment = signal[:, start:end]
        start = start + hop_length
        end = end + hop_length
        segments.append(segment)
    return torch.stack(segments, dim = 1)

def predict_and_overlap(model, full_signal, segment_samples = 6 * 44100, overlap = 0.5, n_fft = 2048, hop_length = 512, win_length = 2048, show_progress = True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.eval().to(device)
    full_signal = full_signal.to(device)
    length = full_signal.shape[1]
    full_signal = zero_pad(full_signal)
    segments = split_to_segments(full_signal)
    num_channels, num_segments, segment_samples = segments.shape
    output = torch.zeros_like(full_signal)
    hop_samples = int(segment_samples * (1 - overlap))
    start, end = 0, segment_samples
    with torch.no_grad():
        with tq.trange(num_segments, desc="Segment ") as segments_tq:
            for num_segment in segments_tq:
                input_segment = segments[:,num_segment,:].unsqueeze(0)
                mask_prediction, input_spectrogram = model(input_segment)
                stft_prediction = (mask_prediction * input_spectrogram)
                stft_prediction = torch.complex(stft_prediction[0,:,:,:,0], stft_prediction[0,:,:,:,1])
                signal_prediction = torch.istft(stft_prediction, n_fft = n_fft, hop_length = hop_length, win_length = win_length, length = segment_samples)
                output[:,start:end] += signal_prediction
                start += hop_samples
                end += hop_samples
    return output, full_signal

def eval_dir(model, dir_path = 'test/', out_path = 'outputs/', full_test_mode = False):
    filenames = os.listdir(dir_path)
    cSDRs = []
    for filename in filenames:
        print("processing " + filename)
        mixture, sr1 = torchaudio.load(dir_path + filename + '/mixture.wav')
        source, sr2 = torchaudio.load(dir_path + filename + '/vocals.wav')
        assert(sr1 == sr2 and sr1 == 44100)
        estimate = predict_and_overlap(model, mixture)
        torchaudio.save(out_path + filename + '_vocal.wav', estimate.cpu().detach(), 44100, channels_first = True,)
        if full_test_mode:
            sdr, isr, sir, sar  = museval.evaluate(source.unsqueeze(0).permute(0, 2, 1).cpu().numpy(), estimate.unsqueeze(0).permute(0, 2, 1).cpu().numpy())
        cSDR = np.mean(np.nanmedian(sdr, axis = 1))
        print("cSDR: ", cSDR)
        cSDRs.append(cSDR)
    return cSDRs


def one_song_from_filepath(filepath, model, source_path = "", offset = 0.0, length = -1, sampling_rate = 44100, force_mono = True, eval = False, full_eval_mode = False, plot_spectrograms = False):
    signal, sr = torchaudio.load(filepath, frame_offset = int(offset * sampling_rate), num_frames = int(length * sampling_rate) if length != -1 else -1)
    scores = []
    if force_mono:
        signal = torch.mean(signal, dim = 0).unsqueeze(0)
    estimate, signal = predict_and_overlap(model, signal)
    torchaudio.save(filepath[:-3] + '_vocals_pred.wav', estimate.cpu().detach(), 44100, channels_first = True,)
    if eval and source_path != "":
        source, sr = torchaudio.load(source_path, frame_offset = int(offset * sampling_rate), num_frames = int(length * sampling_rate) if length != -1 else -1)
        if force_mono:
            source = torch.mean(source, dim = 0).unsqueeze(0)
        source = zero_pad(source).to(signal.device)
        ref_chunks = torch.stack(torch.chunk(source, source.shape[1] // sampling_rate, dim = 1), dim = 0)
        est_chunks = torch.stack(torch.chunk(estimate, estimate.shape[1] // sampling_rate, dim = 1), dim = 0)
        csdr = fast_bss_eval.sdr(ref_chunks, est_chunks, use_cg_iter = 20, clamp_db = 30, load_diag = 1e-5)
        csdr = torch.nanmedian(csdr)
        scores.append(csdr)
        if full_eval_mode:
            sdr, isr, sir, sar  = museval.evaluate(source.unsqueeze(0).permute(0, 2, 1).cpu().numpy(), estimate.unsqueeze(0).permute(0, 2, 1).cpu().numpy())
            cSDR = np.mean(np.nanmedian(sdr, axis = 1))
            scores.append(cSDR)
        if plot_spectrograms:
            mixture_mag_stft = torch.abs(torch.stft(signal.squeeze(0), n_fft = 2048, hop_length = 512, return_complex = True))
            source_mag_stft = torch.abs(torch.stft(source.squeeze(0), n_fft = 2048, hop_length = 512, return_complex = True))
            estimate_mag_stft = torch.abs(torch.stft(estimate.squeeze(0), n_fft = 2048, hop_length = 512, return_complex = True))
            plot_spectrogram(mixture_mag_stft.cpu(), title = "Mixture spectrogram (dB)")
            plot_spectrogram(source_mag_stft.cpu(), title = "Source spectrogram (dB)")
            plot_spectrogram(estimate_mag_stft.cpu(), title = "Predicted source spectrogram (dB)")
        return scores


if __name__ == '__main__':
    # load bsrnn model - load in separate notebook to load bsrnn vs chroma models since the file names are the same and Colab does weird things with imports
    import argparse
    from hparams import hparams_def, hparams_chroma
    from model import Model, ChromaModel
    from trainer import load_model
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, default = 'bsrnn', help = 'bsrnn or sa or fc')
    parser.add_argument('--model_path', type = str, default = 'checkpt_latest.pt', help = 'path to model checkpoint')
    parser.add_argument('--song_path', type = str, default = 'mixture.wav', help = 'path to song to separate')
    parser.add_argument('--source_path', type = str, default = 'vocals.wav', help = 'path to source to evaluate against')
    parser.add_argument('--offset', type = float, default = 0.0, help = 'offset in seconds to start separating from')
    parser.add_argument('--length', type = float, default = 30, help = 'length in seconds to separate')
    parser.add_argument('--eval', type = bool, default = False, help = 'whether to evaluate the separated source')
    parser.add_argument('--full_eval_mode', type = bool, default = False, help = 'whether to evaluate the separated source using museval')
    parser.add_argument('--plot_spectrograms', type = bool, default = False, help = 'whether to plot the spectrograms of the mixture, source, and estimate')
    parser.add_argument('--force_mono', type = bool, default = True, help = 'whether to force the input to be mono')
    args = parser.parse_args()
    if args.model == 'bsrnn':
        model = Model(hparams_def).eval()
    elif args.model == 'sa':
        model = ChromaModel(hparams_chroma, 'attention').eval()
    else:
        model = ChromaModel(hparams_chroma, 'group_fc').eval()
    load_model(args.model_path, model)
    one_song_from_filepath(args.song_path, model, args.source_path, args.offset, args.length, force_mono = args.force_mono, eval = args.eval, full_eval_mode = args.full_eval_mode, plot_spectrograms = args.plot_spectrograms)