import pathlib
import shutil
from pathlib import Path

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import stempeg

from sample_downloader.download import MUSDB_SAMPLE_OUTPUT_PATH

SPECTROGRAM_SAMPLE_OUTPUT_PATH= "{}/sample_data/spectograms"


# --- Average stereo to mono for all signals ---
def stereo_to_mono(signal):
    if signal.ndim == 2 and signal.shape[1] == 2:
        return np.mean(signal, axis=1)
    return signal


# --- Helper function to create normalized mel spectrogram ---
def compute_normalized_mel(y, sr):
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_min = mel_spec_db.min()
    mel_max = mel_spec_db.max()
    mel_norm = (mel_spec_db - mel_min) / (mel_max - mel_min)
    return mel_norm, mel_spec_db


# --- Helper function to plot and save spectrograms without borders ---
def save_spectrogram_image(mel_db, save_path, rate):
    fig, ax = plt.subplots(figsize=(12, 8))
    librosa.display.specshow(mel_db, sr=rate, x_axis='time', y_axis='mel', ax=ax)
    plt.axis('off')
    plt.margins(0)
    plt.subplots_adjust(0, 0, 1, 1)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def convert():
    # Get the current working directory (assuming you run the script from project root)
    project_root = Path.cwd()

    # Define output directory for spectrogram files
    output_dir = Path(SPECTROGRAM_SAMPLE_OUTPUT_PATH.format(project_root))

    # --- Clean output directory ---
    if output_dir.exists():
        shutil.rmtree(output_dir)  # Delete everything inside output_dir
    output_dir.mkdir(parents=True, exist_ok=True)  # Recreate fresh

    # Define the MUSDB samples root directory
    musdb_root = Path(MUSDB_SAMPLE_OUTPUT_PATH.format(project_root))

    # Loop over both train and test datasets
    for split in ['train', 'test']:
        split_path = musdb_root / split

        for file in split_path.glob('*.stem.mp4'):
            print(f"Processing {file}")

            # --- Load audio stems ---
            audio, rate = stempeg.read_stems(str(file))

            # --- Extract relevant parts ---
            mix = audio[0]  # full mix
            drums = audio[1]
            bass = audio[2]
            other = audio[3]
            vocals = audio[4]

            # --- Create instruments by adding drums, bass, and other ---
            instruments = drums + bass + other

            mix = stereo_to_mono(mix)
            vocals = stereo_to_mono(vocals)
            instruments = stereo_to_mono(instruments)

            # --- Compute normalized mel spectrograms and original dB spectrograms ---
            mix_mel_norm, mix_mel_db = compute_normalized_mel(mix, rate)
            vocals_mel_norm, vocals_mel_db = compute_normalized_mel(vocals, rate)
            instruments_mel_norm, instruments_mel_db = compute_normalized_mel(instruments, rate)

            # --- Save normalized mel spectrograms as .npy files ---
            np.save(output_dir / f"{file.stem}_mix.npy", mix_mel_norm)
            np.save(output_dir / f"{file.stem}_vocals.npy", vocals_mel_norm)
            np.save(output_dir / f"{file.stem}_instruments.npy", instruments_mel_norm)

            # --- Save spectrogram plots as .png files ---
            save_spectrogram_image(mix_mel_db, output_dir / f"{file.stem}_mix.png", rate)
            save_spectrogram_image(vocals_mel_db, output_dir / f"{file.stem}_vocals.png", rate)
            save_spectrogram_image(instruments_mel_db, output_dir / f"{file.stem}_instruments.png", rate)

            print(f"Saved .npy and .png for {file.stem}")

if __name__ == "__main__":
    convert()
