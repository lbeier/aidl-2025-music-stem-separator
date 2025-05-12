# u_net_mel/dataset.py (Example structure - adapt to your actual file)
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import random


class ImprovedMelDataset(Dataset):
    def __init__(self, data_dir, chunk_size=128, overlap=0.5, augment=True):
        """
        Args:
            data_dir: Directory containing processed spectrograms
            chunk_size: Number of time frames per chunk
            overlap: Overlap between chunks (0-1)
            augment: Whether to apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.augment = augment

        # Find all mix files
        self.mix_files = list(self.data_dir.glob('*_mix.npy'))

        # Pre-compute chunk indices
        self.chunks = []
        for mix_file in self.mix_files:
            mix_data = np.load(mix_file)
            n_frames = mix_data.shape[1]

            # Calculate chunk indices with overlap
            step = int(chunk_size * (1 - overlap))
            for start in range(0, n_frames - chunk_size, step):
                self.chunks.append((mix_file, start))

        print(
            f"Created dataset with {len(self.chunks)} chunks from {len(self.mix_files)} songs")

    def __len__(self):
        return len(self.chunks)

    def augment_chunk(self, mix_chunk, vocals_chunk):
        """Apply data augmentation to chunks"""
        if random.random() < 0.5:
            # Time stretching
            rate = random.uniform(0.9, 1.1)
            mix_chunk = self._time_stretch(mix_chunk, rate)
            vocals_chunk = self._time_stretch(vocals_chunk, rate)

        if random.random() < 0.5:
            # Frequency masking
            mix_chunk = self._freq_mask(mix_chunk)
            vocals_chunk = self._freq_mask(vocals_chunk)

        if random.random() < 0.5:
            # Time masking
            mix_chunk = self._time_mask(mix_chunk)
            vocals_chunk = self._time_mask(vocals_chunk)

        return mix_chunk, vocals_chunk

    def _time_stretch(self, chunk, rate):
        """Apply time stretching to a chunk"""
        n_frames = chunk.shape[1]
        new_n_frames = int(n_frames * rate)
        stretched = np.zeros((chunk.shape[0], new_n_frames))
        indices = np.linspace(0, n_frames - 1, new_n_frames)
        for i in range(chunk.shape[0]):
            stretched[i] = np.interp(indices, np.arange(n_frames), chunk[i])
        return stretched

    def _freq_mask(self, chunk, max_width=8):
        """Apply frequency masking"""
        width = random.randint(1, max_width)
        start = random.randint(0, chunk.shape[0] - width)
        chunk[start:start+width] = 0
        return chunk

    def _time_mask(self, chunk, max_width=8):
        """Apply time masking"""
        width = random.randint(1, max_width)
        start = random.randint(0, chunk.shape[1] - width)
        chunk[:, start:start+width] = 0
        return chunk

    def __getitem__(self, idx):
        mix_file, start = self.chunks[idx]
        vocals_file = mix_file.parent / \
            mix_file.name.replace('_mix.npy', '_vocals.npy')

        # Load chunks
        mix_chunk = np.load(mix_file)[:, start:start + self.chunk_size]
        vocals_chunk = np.load(vocals_file)[:, start:start + self.chunk_size]

        # Apply augmentation if enabled
        if self.augment:
            mix_chunk, vocals_chunk = self.augment_chunk(
                mix_chunk, vocals_chunk)

        # Ensure output is exactly chunk_size in time dimension
        def fix_length(chunk, chunk_size):
            if chunk.shape[1] > chunk_size:
                return chunk[:, :chunk_size]
            elif chunk.shape[1] < chunk_size:
                pad_width = chunk_size - chunk.shape[1]
                return np.pad(chunk, ((0, 0), (0, pad_width)), mode='constant')
            else:
                return chunk

        mix_chunk = fix_length(mix_chunk, self.chunk_size)
        vocals_chunk = fix_length(vocals_chunk, self.chunk_size)

        # Normalize to [0, 1]
        mix_min, mix_max = mix_chunk.min(), mix_chunk.max()
        vocals_min, vocals_max = vocals_chunk.min(), vocals_chunk.max()

        if mix_max > mix_min:
            mix_norm = (mix_chunk - mix_min) / (mix_max - mix_min)
        else:
            mix_norm = np.zeros_like(mix_chunk)

        if vocals_max > vocals_min:
            vocals_norm = (vocals_chunk - vocals_min) / \
                (vocals_max - vocals_min)
        else:
            vocals_norm = np.zeros_like(vocals_chunk)

        # Convert to tensors
        mix_tensor = torch.from_numpy(mix_norm).float().unsqueeze(0)
        vocals_tensor = torch.from_numpy(vocals_norm).float().unsqueeze(0)

        return mix_tensor, vocals_tensor
