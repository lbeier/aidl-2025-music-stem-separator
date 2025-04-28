import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class MelSpectrogramDataset(Dataset):
    """
    PyTorch Dataset for loading pairs of (mix, vocals) MEL spectrograms.

    This dataset expects .npy files:
    - *_mix_mel.npy : Mix spectrogram (input)
    - *_vocals_mel.npy : Corresponding vocals spectrogram (target)
    """

    def __init__(self, root_dir):
        """
        Args:
            root_dir (str or Path): Directory containing the .npy spectrogram files.
        """
        self.root_dir = Path(root_dir)

        # Collect all mix files matching "*_mix_mel.npy" pattern
        # (Assumes corresponding vocals files exist with similar names)
        self.mel_files = sorted(self.root_dir.glob("*_mix_mel.npy"))

    def __len__(self):
        """
        Return the number of samples available.

        Returns:
            int: Number of mix files (also number of pairs)
        """
        return len(self.mel_files)

    def __getitem__(self, idx):
        """
        Get one (mix, vocals) spectrogram pair.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple(torch.Tensor, torch.Tensor): (mix, vocals) spectrograms as tensors.
        """
        # Get path to mix file
        mix_path = self.mel_files[idx]

        # Infer corresponding vocals file path
        vocals_path = mix_path.with_name(mix_path.name.replace("mix_mel", "vocals_mel"))

        # Load mix and vocals numpy arrays
        mix = np.load(mix_path)  # Shape: (n_mels, time)
        vocals = np.load(vocals_path)  # Shape: (n_mels, time)

        # Convert to PyTorch tensors
        # Add a channel dimension (1, n_mels, time) for CNN input
        mix = torch.tensor(mix, dtype=torch.float32).unsqueeze(0)
        vocals = torch.tensor(vocals, dtype=torch.float32).unsqueeze(0)

        return mix, vocals