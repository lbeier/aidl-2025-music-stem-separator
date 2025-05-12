import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import stempeg
from joblib import Parallel, delayed
import librosa
import soundfile as sf

from sample_downloader.download import MUSDB_SAMPLE_OUTPUT_PATH

# --- Configurable constants ---
SPECTROGRAM_OUTPUT_PATH = "{}/sample_data/spectrograms"
WAVEFORM_OUTPUT_PATH = "{}/sample_data/waveforms"
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
CHUNK_SIZE = 128  # Number of frames per chunk

# --- Audio processing utilities ---
def stereo_to_mono(signal: np.ndarray) -> np.ndarray:
    if signal.ndim == 2 and signal.shape[1] == 2:
        return np.mean(signal, axis=1)
    return signal

def compute_mel(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_norm = np.clip((mel_db + 80) / 80, 0, 1)
    return mel_norm.astype(np.float32), mel_db.astype(np.float32)

def compute_stft(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    magnitude, _ = librosa.magphase(stft)
    stft_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    stft_norm = np.clip((stft_db + 80) / 80, 0, 1)
    return stft_norm.astype(np.float32), stft_db.astype(np.float32)

def save_spectrogram_image(spec_db: np.ndarray, save_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(spec_db, aspect='auto', origin='lower', interpolation='none')
    ax.axis('off')
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def save_waveform_image(signal: np.ndarray, save_path: Path):
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.plot(signal, linewidth=0.5)
    ax.axis('off')
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def chunk_and_save(array: np.ndarray, base_path: Path, name: str):
    num_chunks = array.shape[1] // CHUNK_SIZE
    for i in range(num_chunks):
        chunk = array[:, i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]
        np.save(base_path / f"{name}_chunk_{i:03d}.npy", chunk.astype(np.float32))

# --- Per-file processor ---
def process_file(file: Path, spectrogram_dir: Path, waveform_dir: Path):
    try:
        print(f"Processing {file.name}")

        audio, sr = stempeg.read_stems(str(file))

        mix = stereo_to_mono(audio[0])
        vocals = stereo_to_mono(audio[4])
        instruments = stereo_to_mono(audio[1] + audio[2] + audio[3])

        for name, signal in [("mix", mix), ("vocals", vocals), ("instruments", instruments)]:
            mel_norm, mel_db = compute_mel(signal, sr)
            stft_norm, stft_db = compute_stft(signal, sr)

            base_name = f"{file.stem}_{name}"

            np.save(spectrogram_dir / f"{base_name}_mel.npy", mel_norm)
            np.save(spectrogram_dir / f"{base_name}_mel_db.npy", mel_db)
            np.save(spectrogram_dir / f"{base_name}_stft.npy", stft_norm)
            np.save(spectrogram_dir / f"{base_name}_stft_db.npy", stft_db)

            chunk_and_save(mel_norm, spectrogram_dir, f"{base_name}_mel")
            chunk_and_save(stft_norm, spectrogram_dir, f"{base_name}_stft")

            save_spectrogram_image(mel_db, spectrogram_dir / f"{base_name}_mel.png")
            save_spectrogram_image(stft_db, spectrogram_dir / f"{base_name}_stft.png")

            save_waveform_image(signal, waveform_dir / f"{base_name}_waveform.png")
            sf.write(waveform_dir / f"{base_name}.wav", signal.astype(np.float32), sr)

        print(f"Finished {file.name}")

    except Exception as e:
        print(f"[ERROR] {file}: {e}")

# --- Main processing function ---
def convert():
    project_root = Path.cwd()

    spectrogram_dir = Path(SPECTROGRAM_OUTPUT_PATH.format(project_root))
    waveform_dir = Path(WAVEFORM_OUTPUT_PATH.format(project_root))

    for d in [spectrogram_dir, waveform_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    musdb_root = Path(MUSDB_SAMPLE_OUTPUT_PATH.format(project_root))
    files = []
    for split in ['train', 'test']:
        files.extend((musdb_root / split).glob('*.stem.mp4'))

    if not files:
        print("No stem files found. Check MUSDB_SAMPLE_OUTPUT_PATH.")
        return

    print(f"Found {len(files)} stem files. Starting parallel processing...")

    Parallel(n_jobs=6, backend="threading")(
        delayed(process_file)(file, spectrogram_dir, waveform_dir)
        for file in files
    )

    print("All processing completed successfully.")

if __name__ == "__main__":
    convert()
