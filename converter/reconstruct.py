import numpy as np
import librosa
import soundfile as sf
import argparse
from pathlib import Path

# --- Parameters ---
SR = 44100
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128


def load_normalized_mel(path: Path) -> np.ndarray:
    """Load normalized MEL from .npy file and denormalize to dB scale."""
    mel_norm = np.load(path).astype(np.float32)
    mel_db = mel_norm * 80.0 - 80.0  # assuming normalization from [-80, 0] dB
    mel_power = librosa.db_to_power(mel_db)
    return mel_power


def mel_to_wav(mel_spec: np.ndarray, output_path: Path):
    """Convert MEL spectrogram to audio and save as .wav."""
    audio = librosa.feature.inverse.mel_to_audio(
        mel_spec,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    sf.write(output_path, audio, SR)
    print(f"Saved reconstructed audio to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mel", type=str, required=True, help="Path to normalized mel.npy file")
    parser.add_argument("--out", type=str, required=True, help="Output path for reconstructed .wav file")
    args = parser.parse_args()

    mel_path = Path(args.mel)
    output_path = Path(args.out)

    mel_power = load_normalized_mel(mel_path)
    mel_to_wav(mel_power, output_path)


if __name__ == "__main__":
    main()