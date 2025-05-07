# u_net_mel/dataset.py (Example structure - adapt to your actual file)
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import re # Import regex for parsing filenames

class MelSpectrogramDataset(Dataset):
    def __init__(self, spectrogram_dir):
        """
        Args:
            spectrogram_dir (string): Directory with all chunked .npy spectrograms.
        """
        self.spectrogram_dir = Path(spectrogram_dir)
        self.chunk_files = [] # List to store paths to mix chunks

        # Find all mix MEL chunk files and build the list
        # This implicitly defines the dataset size and allows mapping index to file
        for filepath in self.spectrogram_dir.glob('*_mix_mel_chunk_*.npy'):
            self.chunk_files.append(filepath)

        if not self.chunk_files:
            raise FileNotFoundError(f"No *mix_mel_chunk_*.npy files found in {spectrogram_dir}")

        print(f"Found {len(self.chunk_files)} MEL spectrogram chunks.")

    def __len__(self):
        return len(self.chunk_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        mix_chunk_path = self.chunk_files[idx]

        # Construct the corresponding vocals chunk path from the mix chunk path
        # Assumes filenames like: TRACK_mix_mel_chunk_0000.npy -> TRACK_vocals_mel_chunk_0000.npy
        vocals_chunk_path_str = mix_chunk_path.name.replace("_mix_mel_", "_vocals_mel_")
        vocals_chunk_path = self.spectrogram_dir / vocals_chunk_path_str

        if not vocals_chunk_path.exists():
             # Handle cases where a vocal chunk might be missing if generation failed partially
             # Option 1: Skip this index (might cause issues if not handled carefully)
             # Option 2: Return None or raise error (better handled by dataloader collate/worker)
             # Option 3: Load a zero tensor (simpler for now)
             print(f"Warning: Missing vocal chunk: {vocals_chunk_path}. Returning zeros.")
             # Determine shape from mix chunk
             mix_spec = np.load(mix_chunk_path)
             vocals_spec = np.zeros_like(mix_spec)
        else:
             mix_spec = np.load(mix_chunk_path)
             vocals_spec = np.load(vocals_chunk_path)

        # Add channel dimension (C, H, W) - PyTorch expects channels first
        mix_spec = np.expand_dims(mix_spec, axis=0)
        vocals_spec = np.expand_dims(vocals_spec, axis=0)

        # Convert to torch tensors
        mix_tensor = torch.from_numpy(mix_spec.astype(np.float32))
        vocals_tensor = torch.from_numpy(vocals_spec.astype(np.float32))

        return mix_tensor, vocals_tensor
