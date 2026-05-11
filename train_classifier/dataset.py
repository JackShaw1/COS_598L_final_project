import numpy as np
import torch
from torch.utils.data import Dataset

"""
This script encodes a sample for our classifier.
If less than the maximum number of atoms (100) are present in the pocket, the sample is zero-padded to ensure all samples have the same dimensions.
"""

FEATURE_DIM = 29
N_RESIDUE = 21
N_ELEMENT = 5

class PocketDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        coords, residue_indices, element_indices = self.samples[idx]

        # Create empty one-hot encoding arrays for the residue and element
        residue_one_hot = np.zeros((len(coords), N_RESIDUE), dtype=np.float32)
        element_one_hot = np.zeros((len(coords), N_ELEMENT), dtype=np.float32)

        # Fill in one-hot encodings with real identities
        residue_one_hot[np.arange(len(coords)), residue_indices] = 1.0
        element_one_hot[np.arange(len(coords)), element_indices] = 1.0

        # Create a final array for each atom
        atom_features = np.concatenate([coords, residue_one_hot, element_one_hot], axis=1)
        # Determine whether zero padding or downsampling is needed to ensure equal dimensions for each sample
        atom_features = pad_or_sample_atoms(atom_features)

        x = torch.from_numpy(atom_features.T)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


def pad_or_sample_atoms(atom_features):

    # Determine whether zero padding is needed
    # If there are more than 100 atoms present, we randomly sample 100 atoms
    n_atoms = len(atom_features)
    if n_atoms > 100:
        selected_atoms = np.random.choice(n_atoms, 100, replace=False)
        return atom_features[selected_atoms].astype(np.float32)

    # Zero padding to ensure each sample is the same dimensions as others, even if there are not as many atoms present
    padding = np.zeros((100 - n_atoms, FEATURE_DIM), dtype=np.float32)
    return np.concatenate([atom_features, padding], axis=0).astype(np.float32)
