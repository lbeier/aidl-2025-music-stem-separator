import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class StftSpectrogramDataset(Dataset):
    """
    PyTorch Dataset for loading triplets of (mix, vocals, instruments) chunked STFT spectrograms.

    This dataset expects three types of .npy files inside a directory and its subdirectories:
    - *_mix_chunk_*.npy : Chunked STFT spectrogram of the mixture
    - *_vocals_chunk_*.npy : Chunked STFT spectrogram of vocals only
    - *_instruments_chunk_*.npy : Chunked STFT spectrogram of instruments only
    """

    def __init__(self, spectrogram_dir):
        """
        Args:
            spectrogram_dir (str or Path): Path to directory containing chunked STFT .npy files.
        """
        self.spectrogram_dir = Path(spectrogram_dir)

        # Find and sort all files matching "**/*_mix_chunk_*.npy" pattern recursively
        self.chunk_files = sorted(self.spectrogram_dir.glob("**/*_mix_chunk_*.npy"))

        if not self.chunk_files:
            raise FileNotFoundError(f"No *mix_chunk_*.npy files found in {spectrogram_dir}")

        print(f"Found {len(self.chunk_files)} STFT spectrogram chunks.")

    def __len__(self):
        """
        Return total number of samples (chunks) in the dataset.

        Returns:
            int: Number of (mix, vocals, instruments) chunk triplets found
        """
        return len(self.chunk_files)

    def __getitem__(self, idx):
        """
        Retrieve a (mix, vocals, instruments) chunk triplet by index.

        Args:
            idx (int): Index of the sample chunk to retrieve.

        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor):
                - mix: Tensor of shape (1, freq_bins, chunk_time_steps)
                - vocals: Tensor of shape (1, freq_bins, chunk_time_steps)
                - instruments: Tensor of shape (1, freq_bins, chunk_time_steps)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        mix_chunk_path = self.chunk_files[idx]

        # Infer corresponding vocals and instruments chunk files by replacing "_mix_chunk_" with respective suffixes
        vocals_chunk_path_str = mix_chunk_path.name.replace("_mix_chunk_", "_vocals_chunk_")
        instruments_chunk_path_str = mix_chunk_path.name.replace("_mix_chunk_", "_instruments_chunk_")
        vocals_chunk_path = mix_chunk_path.parent / vocals_chunk_path_str
        instruments_chunk_path = mix_chunk_path.parent / instruments_chunk_path_str

        # Load the .npy files, use zeros if any are missing
        if not vocals_chunk_path.exists():
            print(f"Warning: Missing vocal chunk: {vocals_chunk_path}. Returning zeros.")
        if not instruments_chunk_path.exists():
            print(f"Warning: Missing instruments chunk: {instruments_chunk_path}. Returning zeros.")

        mix_spec = np.load(mix_chunk_path)
        vocals_spec = np.load(vocals_chunk_path) if vocals_chunk_path.exists() else np.zeros_like(mix_spec)
        instruments_spec = (
            np.load(instruments_chunk_path) if instruments_chunk_path.exists() else np.zeros_like(mix_spec)
        )

        # Add channel dimension (unsqueeze at dimension 0) → (channels, freq, time)
        mix_spec = np.expand_dims(mix_spec, axis=0)
        vocals_spec = np.expand_dims(vocals_spec, axis=0)
        instruments_spec = np.expand_dims(instruments_spec, axis=0)

        # Convert numpy arrays to PyTorch tensors
        mix = torch.tensor(mix_spec, dtype=torch.float32)
        vocals = torch.tensor(vocals_spec, dtype=torch.float32)
        instruments = torch.tensor(instruments_spec, dtype=torch.float32)

        return mix, vocals, instruments
