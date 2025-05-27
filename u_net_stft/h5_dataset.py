import torch
from torch.utils.data import Dataset
import h5py
import numpy as np


class H5SpectrogramDataset(Dataset):
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform
        with h5py.File(h5_path, 'r') as f:
            # Asumimos que todas las keys están en 'mix' y existen en los otros grupos
            self.keys = list(f['mix'].keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        with h5py.File(self.h5_path, 'r') as f:
            mix = f['mix'][key][()]
            vocals = f['vocals'][key][()]
            instruments = f['instruments'][key][()]
        # Añade canal (1, F, T)
        mix = np.expand_dims(mix, 0)
        vocals = np.expand_dims(vocals, 0)
        instruments = np.expand_dims(instruments, 0)
        # A tensor
        mix = torch.tensor(mix, dtype=torch.float32)
        vocals = torch.tensor(vocals, dtype=torch.float32)
        instruments = torch.tensor(instruments, dtype=torch.float32)
        if self.transform:
            mix, vocals, instruments = self.transform(mix, vocals, instruments)
        return mix, vocals, instruments
