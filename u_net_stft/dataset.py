import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class StftSpectrogramDataset(Dataset):
    """
    PyTorch Dataset for loading pairs of (mix, vocals) STFT spectrograms.

    This dataset expects two types of .npy files inside a directory:
    - *_mix_stft.npy : STFT spectrogram of the mixture (input to model)
    - *_vocals_stft.npy : STFT spectrogram of vocals only (target for model)
    """

    def __init__(self, root_dir):
        """
        Args:
            root_dir (str or Path): Path to directory containing STFT .npy files.
        """
        self.root_dir = Path(root_dir)  # Convert root_dir into a Path object for easier path manipulations

        # Find and sort all files matching "*_mix_stft.npy" pattern
        # These will serve as the inputs (mix spectrograms)
        self.stft_files = sorted(self.root_dir.glob("*_mix_stft.npy"))

    def __len__(self):
        """
        Return total number of samples in the dataset.

        Returns:
            int: Number of (mix, vocals) pairs found
        """
        return len(self.stft_files)

    def __getitem__(self, idx):
        """
        Retrieve a (mix, vocals) pair by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple(torch.Tensor, torch.Tensor):
                - mix: Tensor of shape (1, freq_bins, time_steps)
                - vocals: Tensor of shape (1, freq_bins, time_steps)
        """
        # Path to mix STFT file
        mix_path = self.stft_files[idx]

        # Infer corresponding vocals STFT file by replacing "mix_stft" with "vocals_stft"
        vocals_path = mix_path.with_name(mix_path.name.replace("mix_stft", "vocals_stft"))

        # Load the .npy files (2D arrays: frequency bins x time frames)
        mix = np.load(mix_path)
        vocals = np.load(vocals_path)

        # Convert numpy arrays to PyTorch tensors
        # Add a channel dimension (unsqueeze at dimension 0) → (channels, freq, time)
        mix = torch.tensor(mix, dtype=torch.float32).unsqueeze(0)
        vocals = torch.tensor(vocals, dtype=torch.float32).unsqueeze(0)

        return mix, vocals