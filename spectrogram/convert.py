import pathlib
from pathlib import Path

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import stempeg

from sample_downloader.download import MUSDB_SAMPLE_OUTPUT_PATH

SPECTROGRAM_SAMPLE_OUTPUT_PATH= "{}/sample_data/spectograms"

def convert():
    # Get the current working directory (assuming you run the script from project root)
    project_root = Path.cwd()

    # Define output directory for spectrogram files
    output_dir = Path(SPECTROGRAM_SAMPLE_OUTPUT_PATH.format(project_root))

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define the MUSDB samples root directory
    musdb_root = Path(MUSDB_SAMPLE_OUTPUT_PATH.format(project_root))

    # Loop over both train and test datasets
    for split in ['train', 'test']:
        split_path = musdb_root / split

        # Find all .stem.mp4 files inside the split directory
        for file in split_path.glob('*.stem.mp4'):
            print(f"Processing {file}")

            # --- Load audio stems ---
            # stempeg.read_stems loads all stems into a numpy array
            # shape: (num_stems, samples, channels)
            audio, rate = stempeg.read_stems(str(file))

            # --- Mix all stems together into a single audio track ---
            mix = np.mean(audio, axis=0)  # Collapse across stems

            # --- If stereo (2 channels), average them to mono ---
            if mix.ndim == 2 and mix.shape[1] == 2:
                mix = np.mean(mix, axis=1)

            # --- Create MEL spectrogram ---
            # Converts waveform into Mel-scaled spectrogram (human-like frequency perception)
            mel_spec = librosa.feature.melspectrogram(y=mix, sr=rate, n_mels=128)

            # --- Convert power spectrogram (amplitude^2) to decibel (dB) units ---
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # --- Normalize mel spectrogram to [0,1] range ---
            # Important for CNN training: smaller, bounded inputs make models train faster and more stable
            mel_spec_db_min = mel_spec_db.min()
            mel_spec_db_max = mel_spec_db.max()
            mel_spec_db_norm = (mel_spec_db - mel_spec_db_min) / (mel_spec_db_max - mel_spec_db_min)

            # --- Save the normalized Mel spectrogram as a .npy file ---
            # This is what the CNN will actually use as input features
            npy_output_file = output_dir / f"{file.stem}_mel_spectrogram.npy"
            np.save(npy_output_file, mel_spec_db_norm)

            # --- (Optional) Save a plot for human inspection ---
            # Plot original dB mel spectrogram (not normalized) for easier visual checks
            plt.figure(figsize=(12, 8))
            librosa.display.specshow(
                mel_spec_db, sr=rate, x_axis='time', y_axis='mel'
            )
            plt.colorbar(format='%+2.0f dB')  # Color bar shows dB levels
            plt.title(f'Mel Spectrogram - {file.stem}')
            plt.tight_layout()

            # Save plot as PNG image
            png_output_file = output_dir / f"{file.stem}_mel_spectrogram.png"
            plt.savefig(png_output_file)
            plt.close()


if __name__ == "__main__":
    convert()
