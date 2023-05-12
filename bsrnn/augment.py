import torch
import torchaudio
import random
from data import MusHQDataset
from onthefly import PermuteSourcesDataset
from torch.utils.data import Dataset
from torch_audiomentations import Compose, Gain, PolarityInversion, PitchShift, AddColoredNoise, ApplyImpulseResponse
import torch_audiomentations

class MusicAugmentationDataset(Dataset):
    def __init__(self, dataset_path, epoch_size, subset = 'train', split = "train", apply_probability = 0.15, permute = True, sample_rate=44100):
        self.underlying_dataset = MusHQDataset(dataset_path, subset = subset, split = split, epoch_size = epoch_size)
        self.dir_filenames = self.underlying_dataset.dir_filenames
        self.cutoff = self.underlying_dataset.cutoff
        self.root = self.underlying_dataset.root
        if permute:
            self.underlying_dataset = PermuteSourcesDataset(self.underlying_dataset)
        self.sample_rate = sample_rate
        self.apply_prob = apply_probability
        # Initialize the augmentation callable
        self.source_augmentation = Compose(
            transforms=[
                Gain(min_gain_in_db=-8.0, max_gain_in_db=8.0, p=0.5),
                PolarityInversion(p=0.25),
                PitchShift(-4, 4, p = 0.3, sample_rate = 44100)
            ]
        )
        self.accompaniment_augmentation = Compose(
            transforms = [
                Gain(min_gain_in_db=-8.0, max_gain_in_db=8.0, p=0.5),
                PolarityInversion(p=0.25),
                PitchShift(-4, 4, p = 0.3, sample_rate = 44100),
                AddColoredNoise(20,60,p=0.2)
            ]
        )
        

    def __len__(self):
        return len(self.underlying_dataset)

    def __getitem__(self, index):
        # Get the mixture and source from the underlying dataset
        mixture, source = self.underlying_dataset[index]
        # Apply the augmentations only to the source
        modified_source = self.source_augmentation(source.unsqueeze(0), sample_rate=self.sample_rate).squeeze(0)

        accompaniment = mixture - source  

        source_effects = []
        accompaniment_effects = []
        
        modified_accompaniment = self.accompaniment_augmentation(accompaniment.unsqueeze(0), sample_rate = self.sample_rate).squeeze(0)
        
        if random.random() < self.apply_prob:
        # Choose random values for reverberance, HF-damping, and room-scale
            reverberance = random.uniform(30, 70)
            reverberance_source = random.uniform(reverberance - 20, reverberance + 10)
            hf_damping = random.uniform(30, 70)
            room_scale = random.uniform(20, 90)
            room_scale_source = random.uniform(room_scale - 20, room_scale + 10)

            
            # Apply reverb effect to the source
            source_effects.extend([
                ["reverb", f"{reverberance_source}", f"{hf_damping}", f"{room_scale_source}"]
            ])
            accompaniment_effects.extend([
                ["reverb", f"{reverberance}", f"{hf_damping}", f"{room_scale}"]
            ])
            
        # Low-pass filter
        if random.random() < self.apply_prob:
            cutoff_freq = random.uniform(300, self.sample_rate // 2)
            accompaniment_effects.extend([["lowpass", "-1", f"{cutoff_freq}"]])

        # High-pass filter
        if random.random() < self.apply_prob:
            cutoff_freq = random.uniform(20, self.sample_rate // 4)
            accompaniment_effects.extend([["highpass", "-1", f"{cutoff_freq}"]])
        # Subtract the original source and add the new source to the mixture
        if source_effects:
            modified_source, _ = torchaudio.sox_effects.apply_effects_tensor(
                modified_source, self.sample_rate, source_effects
            )
            
            # Ensure the output signal has the same length as the input signal
            if modified_source.size(-1) < source.size(-1):
                padding = torch.zeros(source.size(0), source.size(-1) - modified_source.size(-1))
                modified_source = torch.cat([modified_source, padding], dim=-1)
            else:
                modified_source = modified_source[:, :source.size(-1)]

        if accompaniment_effects:
            modified_accompaniment, _ = torchaudio.sox_effects.apply_effects_tensor(
                modified_accompaniment, self.sample_rate, accompaniment_effects
            )
            
            # Ensure the output signal has the same length as the input signal
            if modified_accompaniment.size(-1) < source.size(-1):
                padding = torch.zeros(accompaniment.size(0), accompaniment.size(-1) - modified_accompaniment.size(-1))
                modified_accompaniment = torch.cat([modified_accompaniment, padding], dim=-1)
            else:
                modified_accompaniment = modified_accompaniment[:, :accompaniment.size(-1)]
        new_mixture = modified_accompaniment + modified_source

        return mixture, source, new_mixture, modified_source, modified_accompaniment
