import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class StftSpectrogramDataset(Dataset):
    """
    PyTorch Dataset for loading pairs of (mix, vocals) chunked STFT spectrograms.

    This dataset expects two types of .npy files inside a directory:
    - *_mix_stft_chunk_*.npy : Chunked STFT spectrogram of the mixture
    - *_vocals_stft_chunk_*.npy : Chunked STFT spectrogram of vocals only
    """

    def __init__(self, spectrogram_dir):
        """
        Args:
            spectrogram_dir (str or Path): Path to directory containing chunked STFT .npy files.
        """
        self.spectrogram_dir = Path(spectrogram_dir)

        # Find and sort all files matching "*_mix_stft_chunk_*.npy" pattern
        self.chunk_files = sorted(self.spectrogram_dir.glob("*_mix_stft_chunk_*.npy"))

        if not self.chunk_files:
            raise FileNotFoundError(f"No *mix_stft_chunk_*.npy files found in {spectrogram_dir}")

        print(f"Found {len(self.chunk_files)} STFT spectrogram chunks.")


    def __len__(self):
        """
        Return total number of samples (chunks) in the dataset.

        Returns:
            int: Number of (mix, vocals) chunk pairs found
        """
        return len(self.chunk_files)

    def __getitem__(self, idx):
        """
        Retrieve a (mix, vocals) chunk pair by index.

        Args:
            idx (int): Index of the sample chunk to retrieve.

        Returns:
            tuple(torch.Tensor, torch.Tensor):
                - mix: Tensor of shape (1, freq_bins, chunk_time_steps)
                - vocals: Tensor of shape (1, freq_bins, chunk_time_steps)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        mix_chunk_path = self.chunk_files[idx]

        # Infer corresponding vocals chunk file by replacing "_mix_stft_" with "_vocals_stft_"
        vocals_chunk_path_str = mix_chunk_path.name.replace("_mix_stft_", "_vocals_stft_")
        vocals_chunk_path = self.spectrogram_dir / vocals_chunk_path_str

        # Load the .npy files
        if not vocals_chunk_path.exists():
            print(f"Warning: Missing vocal chunk: {vocals_chunk_path}. Returning zeros.")
            mix_spec = np.load(mix_chunk_path)
            vocals_spec = np.zeros_like(mix_spec) # Create zero array with same shape
        else:
            mix_spec = np.load(mix_chunk_path)
            vocals_spec = np.load(vocals_chunk_path)

        # Add channel dimension (unsqueeze at dimension 0) → (channels, freq, time)
        mix_spec = np.expand_dims(mix_spec, axis=0)
        vocals_spec = np.expand_dims(vocals_spec, axis=0)

        # Convert numpy arrays to PyTorch tensors
        mix = torch.tensor(mix_spec, dtype=torch.float32)
        vocals = torch.tensor(vocals_spec, dtype=torch.float32)

        return mix, vocals