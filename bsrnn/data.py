from torch.utils.data import DataLoader, Dataset
import torch
import torchaudio
import random
import os


def check_highest_energy(audio, num_chunks = 10, threshold = 4, ratio = 0.15):
    samples = audio.shape[2]
    chunk_samples = samples // num_chunks
    high_energy_chunks = 0

    if torch.mean(torch.pow(audio, 2)) < 0.0005:
        return False
      
    for i in range(num_chunks):
        chunk = audio[:, :, i * chunk_samples: (i+1) * chunk_samples]
        source = chunk[1, :, :]
        mixture = chunk[0, :, :]

        source_energy = torch.sum(torch.pow(source, 2))
        mixture_energy = torch.sum(torch.pow(mixture, 2))
        
        if source_energy / mixture_energy > ratio:
            high_energy_chunks += 1
    
    return high_energy_chunks >= threshold

class MusHQDataset(Dataset):
    def __init__(self,  musdb_root = 'musdb18hq', epoch_size = 1000, train_validation_cutoff_index = 90, sources = ['mixture', 'vocals'], split='train', subset='train', sampling_rate = 44100, is_wav = False, segment_length = 6, discard_low_energy = True):
        self.sources = sources
        self.root = musdb_root + '/' + subset + '/'
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.segment_samples = self.sampling_rate * self.segment_length
        self.dir_filenames = os.listdir(self.root)
        self.discard_low_energy = discard_low_energy
        self.cutoff = train_validation_cutoff_index
        self.epoch_size = epoch_size

        if subset == 'train' and split == 'train':
            self.filenames = self.dir_filenames[:train_validation_cutoff_index]
        elif subset == 'train' and split == 'validation':
            self.filenames = self.dir_filenames[train_validation_cutoff_index:]
        self.num_files = len(self.filenames)


    def __getitem__(self, index):
        filename = self.root + self.filenames[index % self.num_files]
        metadata = torchaudio.info(filename + '/mixture.wav')
        sr, num_frames = metadata.sample_rate, metadata.num_frames
        high_energy = False
        depth = 0
        highest_energy_audio = torch.zeros(self.segment_samples)
        while not high_energy and depth < 10:
            frame_offset = random.randint(0, num_frames - self.segment_samples / 2)
            audio = torch.stack([torchaudio.load(filename + '/' + source + '.wav', frame_offset = frame_offset, num_frames = self.segment_samples)[0] for source in self.sources])
            segment_samples = audio.shape[2]
            to_pad = self.segment_samples - segment_samples
            pad_tensor = torch.zeros(audio.shape[0], audio.shape[1], to_pad)
            padded_audio = torch.cat([audio, pad_tensor], dim = 2)
            if padded_audio.pow(2).sum() > highest_energy_audio.pow(2).sum():
                highest_energy_audio = padded_audio
            high_energy = True if not self.discard_low_energy else check_highest_energy(padded_audio)
            depth += 1
        return highest_energy_audio[0,:,:], highest_energy_audio[1,:,:]

    def __len__(self):
        return self.epoch_size


class PreprocessedDataset(Dataset):
    def __init__(self, data_directory, sample_rate=44100):
        self.data_directory = data_directory
        self.sample_rate = sample_rate
        self.song_names = os.listdir(self.data_directory)
        self.file_pairs = []

        for song_name in self.song_names:
            song_dir = os.path.join(self.data_directory, song_name)
            mixtures_files = [file for file in os.listdir(song_dir) if file.startswith("mixture")]
            sources_files = [file for file in os.listdir(song_dir) if file.startswith("vocals")]

            mixtures_files.sort()
            sources_files.sort()

            for mixture_file, source_file in zip(mixtures_files, sources_files):
                mixture_path = os.path.join(song_dir, mixture_file)
                source_path = os.path.join(song_dir, source_file)
                self.file_pairs.append((mixture_path, source_path))

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        mixture_path, source_path = self.file_pairs[idx]
        mixture, _ = torchaudio.load(mixture_path, num_frames=self.sample_rate * 6)
        source, _ = torchaudio.load(source_path, num_frames=self.sample_rate * 6)
        return mixture, source
