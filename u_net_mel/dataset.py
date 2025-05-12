import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import re

class MelSpectrogramDataset(Dataset):
    def __init__(self, spectrogram_dir):
        """
        Args:
            spectrogram_dir (string): Directory with all chunked .npy spectrograms.
        """
        self.spectrogram_dir = Path(spectrogram_dir)
        self.chunk_files = sorted(self.spectrogram_dir.glob("*_mix_mel_chunk_*.npy"))

        if not self.chunk_files:
            raise FileNotFoundError(f"No *mix_mel_chunk_*.npy files found in {spectrogram_dir}")

        print(f"Found {len(self.chunk_files)} MEL spectrogram chunks.")

    def __len__(self):
        return len(self.chunk_files)

    def __getitem__(self, idx):
        mix_path = self.chunk_files[idx]
        chunk_id = self._chunk_index(mix_path)
        base = str(mix_path.name).replace(f"_mix_mel_chunk_{chunk_id}.npy", "")

        vocals_path = self.spectrogram_dir / f"{base}_vocals_mel_chunk_{chunk_id}.npy"
        instruments_path = self.spectrogram_dir / f"{base}_instruments_mel_chunk_{chunk_id}.npy"

        if not vocals_path.exists() or not instruments_path.exists():
            raise FileNotFoundError(f"Missing chunk files for: {mix_path.name}")

        mix = np.load(mix_path)
        vocals = np.load(vocals_path)
        instruments = np.load(instruments_path)

        return (
            torch.tensor(mix, dtype=torch.float32).unsqueeze(0),
            torch.tensor(vocals, dtype=torch.float32).unsqueeze(0),
            torch.tensor(instruments, dtype=torch.float32).unsqueeze(0),
        )

    def _chunk_index(self, path):
        match = re.search(r"chunk_(\d+)", path.stem)
        return match.group(1) if match else "000"