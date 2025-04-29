import shutil
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import stempeg
from joblib import Parallel, delayed

from sample_downloader.download import MUSDB_SAMPLE_OUTPUT_PATH

# --- Configurable constants ---
SPECTROGRAM_OUTPUT_PATH = "{}/sample_data/spectrograms"  # (fixed spelling!)
WAVEFORM_OUTPUT_PATH = "{}/sample_data/waveforms"
N_MELS = 128  # Number of Mel bands
N_FFT = 2048  # FFT window size for STFT
HOP_LENGTH = 512  # Hop length for STFT


# --- Audio processing utilities ---
def stereo_to_mono(signal: np.ndarray) -> np.ndarray:
    """Convert stereo signal to mono by averaging channels."""
    if signal.ndim == 2 and signal.shape[1] == 2:
        return np.mean(signal, axis=1)
    return signal


def compute_normalized_mel(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized Mel spectrogram and dB Mel spectrogram."""
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
    return mel_norm, mel_db


def compute_normalized_stft(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized STFT spectrogram and dB STFT spectrogram."""
    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    magnitude, _ = librosa.magphase(stft)
    stft_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    stft_norm = (stft_db - stft_db.min()) / (stft_db.max() - stft_db.min())
    return stft_norm, stft_db


def save_spectrogram_image(spec_db: np.ndarray, save_path: Path, sr: int):
    """Plot and save spectrogram image without borders."""
    fig, ax = plt.subplots(figsize=(12, 8))
    librosa.display.specshow(spec_db, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', ax=ax)
    plt.axis('off')
    plt.margins(0)
    plt.subplots_adjust(0, 0, 1, 1)
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def save_waveform_image(signal: np.ndarray, save_path: Path, sr: int):
    """Plot and save waveform image without borders."""
    fig, ax = plt.subplots(figsize=(12, 3))
    librosa.display.waveshow(signal, sr=sr, ax=ax)
    plt.axis('off')
    plt.margins(0)
    plt.subplots_adjust(0, 0, 1, 1)
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


# --- Per-file processor ---
def process_file(file: Path, spectrogram_dir: Path, waveform_dir: Path):
    """Process one audio stem file: generate and save MEL, STFT, waveform outputs."""
    try:
        print(f"Processing {file.name}")

        audio, sr = stempeg.read_stems(str(file))

        # Extract stems and convert to mono
        mix = stereo_to_mono(audio[0])
        vocals = stereo_to_mono(audio[4])
        instruments = stereo_to_mono(audio[1] + audio[2] + audio[3])

        # MEL spectrograms
        mix_mel_norm, mix_mel_db = compute_normalized_mel(mix, sr)
        vocals_mel_norm, vocals_mel_db = compute_normalized_mel(vocals, sr)
        instruments_mel_norm, instruments_mel_db = compute_normalized_mel(instruments, sr)

        # STFT spectrograms
        mix_stft_norm, mix_stft_db = compute_normalized_stft(mix, sr)
        vocals_stft_norm, vocals_stft_db = compute_normalized_stft(vocals, sr)
        instruments_stft_norm, instruments_stft_db = compute_normalized_stft(instruments, sr)

        # Save npy normalized arrays
        np.save(spectrogram_dir / f"{file.stem}_mix_mel.npy", mix_mel_norm)
        np.save(spectrogram_dir / f"{file.stem}_vocals_mel.npy", vocals_mel_norm)
        np.save(spectrogram_dir / f"{file.stem}_instruments_mel.npy", instruments_mel_norm)

        np.save(spectrogram_dir / f"{file.stem}_mix_stft.npy", mix_stft_norm)
        np.save(spectrogram_dir / f"{file.stem}_vocals_stft.npy", vocals_stft_norm)
        np.save(spectrogram_dir / f"{file.stem}_instruments_stft.npy", instruments_stft_norm)

        # Save spectrogram images
        save_spectrogram_image(mix_mel_db, spectrogram_dir / f"{file.stem}_mix_mel.png", sr)
        save_spectrogram_image(vocals_mel_db, spectrogram_dir / f"{file.stem}_vocals_mel.png", sr)
        save_spectrogram_image(instruments_mel_db, spectrogram_dir / f"{file.stem}_instruments_mel.png", sr)

        save_spectrogram_image(mix_stft_db, spectrogram_dir / f"{file.stem}_mix_stft.png", sr)
        save_spectrogram_image(vocals_stft_db, spectrogram_dir / f"{file.stem}_vocals_stft.png", sr)
        save_spectrogram_image(instruments_stft_db, spectrogram_dir / f"{file.stem}_instruments_stft.png", sr)

        # Save waveform images
        save_waveform_image(mix, waveform_dir / f"{file.stem}_mix_waveform.png", sr)
        save_waveform_image(vocals, waveform_dir / f"{file.stem}_vocals_waveform.png", sr)
        save_waveform_image(instruments, waveform_dir / f"{file.stem}_instruments_waveform.png", sr)

        print(f"Finished {file.name}")

    except Exception as e:
        print(f"Error processing {file.name}: {e}")


# --- Main processing function ---
def convert():
    """Main entry point to process all audio stems into spectrogram and waveform images."""
    project_root = Path.cwd()

    spectrogram_dir = Path(SPECTROGRAM_OUTPUT_PATH.format(project_root))
    waveform_dir = Path(WAVEFORM_OUTPUT_PATH.format(project_root))

    # Clean and recreate output folders
    for d in [spectrogram_dir, waveform_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # Locate dataset
    musdb_root = Path(MUSDB_SAMPLE_OUTPUT_PATH.format(project_root))
    files = []
    for split in ['train', 'test']:
        files.extend((musdb_root / split).glob('*.stem.mp4'))

    if not files:
        print("No stem files found. Check MUSDB_SAMPLE_OUTPUT_PATH.")
        return

    print(f"Found {len(files)} stem files. Starting parallel processing...")

    # Parallel execution across all CPU cores
    Parallel(n_jobs=-1)(
        delayed(process_file)(file, spectrogram_dir, waveform_dir)
        for file in files
    )

    print("All processing completed successfully.")


if __name__ == "__main__":
    convert()