import torch
from torch.utils.data import Dataset
import random

class PermuteSourcesDataset(Dataset):
    def __init__(self, underlying_dataset, probability = 0.4):
        self.underlying_dataset = underlying_dataset
        self.permute_probability = probability
    
    def __len__(self):
        return len(self.underlying_dataset)

    def __getitem__(self, idx):
        mixture1, source1 = self.underlying_dataset[idx]
        # Sample 2 random indices
        if random.random() < self.permute_probability:
            index = random.sample(range(len(self.underlying_dataset)), 1)
      
            # Get the mixtures and sources for the sampled indices
            mixture2, source2 = self.underlying_dataset[index[0]]

            # Get the accompaniment from the first song
            accompaniment1 = mixture1 - source1

            # Create a new mixture by combining the accompaniment from the first song and the vocals from the second
            new_mixture = accompaniment1 + source2
            return new_mixture, source2
        # The new source is the vocals from the second song
        else:
            return mixture1, source1
