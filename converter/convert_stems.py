# --- First: Install dependencies (in a separate Colab cell) ---
!pip install stempeg librosa matplotlib

# --- Then: Python script starts here ---
import shutil
from pathlib import Path
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import stempeg
import os

# Use non-GUI backend
matplotlib.use("Agg")

# --- Configurable constants ---
SPECTROGRAM_OUTPUT_PATH = "spectrograms"
WAVEFORM_OUTPUT_PATH = "waveforms"
N_FFT = 2048
HOP_LENGTH = 512

# Change this to wherever your .stem.mp4 files are stored
MUSDB_SAMPLE_PATH = Path("/content/sample_data/musdb")

# --- Utility functions ---
def stereo_to_mono(signal: np.ndarray) -> np.ndarray:
    return np.mean(signal, axis=1) if signal.ndim == 2 and signal.shape[1] == 2 else signal

def compute_normalized_stft(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    magnitude, _ = librosa.magphase(stft)
    stft_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    stft_norm = (stft_db - stft_db.min()) / (stft_db.max() - stft_db.min())
    return stft_norm, stft_db

def save_spectrogram_image(spec_db: np.ndarray, save_path: Path, sr: int):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(spec_db, aspect="auto", origin="lower", cmap="magma")
    ax.set_title("Spectrogram (dB)")
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def save_waveform_image(signal: np.ndarray, save_path: Path, sr: int):
    fig, ax = plt.subplots(figsize=(12, 3))
    librosa.display.waveshow(signal, sr=sr, ax=ax)
    plt.axis("off")
    plt.margins(0)
    plt.subplots_adjust(0, 0, 1, 1)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def process_file(file: Path, spectrogram_dir: Path, waveform_dir: Path):
    try:
        print(f"Processing {file.name}")
        audio, sr = stempeg.read_stems(str(file))

        mix = stereo_to_mono(audio[0])
        vocals = stereo_to_mono(audio[4])
        instruments = stereo_to_mono(audio[1] + audio[2] + audio[3])

        chunk_size = librosa.time_to_frames(6.0, sr=sr, hop_length=HOP_LENGTH)

        mix_stft_norm, mix_stft_db = compute_normalized_stft(mix, sr)
        vocals_stft_norm, vocals_stft_db = compute_normalized_stft(vocals, sr)
        instruments_stft_norm, instruments_stft_db = compute_normalized_stft(instruments, sr)

        chunk_dir = spectrogram_dir / file.stem.replace(".stem", "")
        chunk_dir.mkdir(parents=True, exist_ok=True)

        def save_chunks(spectrogram, label):
            n_frames = spectrogram.shape[1]
            for i in range(0, n_frames - chunk_size + 1, chunk_size):
                chunk = spectrogram[:, i : i + chunk_size]
                chunk_filename = f"{file.stem}_{label}_chunk_{i // chunk_size:02d}.npy"
                np.save(chunk_dir / chunk_filename, chunk)

        save_chunks(mix_stft_norm, "mix")
        save_chunks(vocals_stft_norm, "vocals")
        save_chunks(instruments_stft_norm, "instruments")

        save_spectrogram_image(mix_stft_db, spectrogram_dir / f"{file.stem}_mix_stft.png", sr)
        save_spectrogram_image(vocals_stft_db, spectrogram_dir / f"{file.stem}_vocals_stft.png", sr)
        save_spectrogram_image(instruments_stft_db, spectrogram_dir / f"{file.stem}_instruments_stft.png", sr)

        save_waveform_image(mix, waveform_dir / f"{file.stem}_mix_waveform.png", sr)
        save_waveform_image(vocals, waveform_dir / f"{file.stem}_vocals_waveform.png", sr)
        save_waveform_image(instruments, waveform_dir / f"{file.stem}_instruments_waveform.png", sr)

        print(f"Finished {file.name}")
    except Exception as e:
        print(f"Error processing {file.name}: {e}")

def convert():
    spectrogram_dir = Path(SPECTROGRAM_OUTPUT_PATH)
    waveform_dir = Path(WAVEFORM_OUTPUT_PATH)

    for d in [spectrogram_dir, waveform_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    files = list(MUSDB_SAMPLE_PATH.glob("**/*.stem.mp4"))

    if not files:
        print("No stem files found. Upload or mount your dataset.")
        return

    print(f"Found {len(files)} stem files. Starting serial processing...")
    for file in files:
        process_file(file, spectrogram_dir, waveform_dir)

    print("All processing completed successfully.")

# Run it
convert()