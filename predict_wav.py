import argparse
import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

# Import STFT model only
from u_net_stft.model import UNetSmall as StftUNet

# Spectrogram parameters (MUST match converter/converter_wav.py)
N_FFT = 2048
HOP_LENGTH = 512
TARGET_SR = 44100


def get_device():
    """Detect the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_audio(filepath, sr=TARGET_SR):
    """Load a WAV file (ensure mono)."""
    # Load as mono directly
    y, loaded_sr = librosa.load(filepath, sr=None, mono=True)
    if loaded_sr != sr:
        print(f"Warning: Resampling audio from {loaded_sr} Hz to {sr} Hz.")
        y = librosa.resample(y, orig_sr=loaded_sr, target_sr=sr)
    return y


def save_audio(y, path, sr=TARGET_SR):
    """Save waveform to a WAV file."""
    if y is None or len(y) == 0 or np.isnan(y).any() or np.isinf(y).any():
        print(f"[ERROR] Invalid audio. Not saving to {path}.")
        return
    sf.write(path, y, sr)


def predict_wav(model_path, input_wav, output_vocals_wav, output_instruments_wav):
    """Predict vocals and instruments from a WAV file using the trained STFT model."""
    device = get_device()
    print(f"Using device: {device}")

    ModelClass = StftUNet
    print("Using STFT model")

    # Load trained model
    model = ModelClass(in_channels=1, out_channels=1).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model state_dict from {model_path}: {e}")
        print("Ensure the model path is correct and the state_dict matches the ModelClass.")
        return
    model.eval()

    # Load input audio
    print(f"Loading audio: {input_wav}")
    y = load_audio(input_wav, sr=TARGET_SR)

    # --- Calculate STFT spectrogram and Normalize (Consistent with converter_wav.py) ---
    print("Calculating STFT spectrogram...")
    stft_result = librosa.stft(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    magnitude, phase = librosa.magphase(stft_result)
    stft_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    min_db = stft_db.min()
    max_db = stft_db.max()
    print(f"  Input STFT dB range: [{min_db:.2f}, {max_db:.2f}]")
    if max_db > min_db:
        spec_norm = (stft_db - min_db) / (max_db - min_db)
    else:
        spec_norm = np.zeros_like(stft_db)

    # Prepare input for model
    spec_tensor = torch.tensor(spec_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Predict the vocals mask/spectrogram
    print("Predicting with model...")
    with torch.no_grad():
        pred_output = model(spec_tensor).squeeze().cpu().numpy()
        pred_vocals_norm = np.clip(pred_output[0], 0, 1)
        pred_instruments_norm = np.clip(pred_output[1], 0, 1)

    print(f"  Predicted vocals norm range: [{pred_vocals_norm.min():.2f}, {pred_vocals_norm.max():.2f}]")
    print(f"  Estimated instruments norm range: [{pred_instruments_norm.min():.2f}, {pred_instruments_norm.max():.2f}]")

    # --- Convert back to waveform using consistent de-normalization ---
    print("Converting predictions back to audio...")
    vocals_audio = None
    instruments_audio = None
    denormalization_range = max_db - min_db

    if denormalization_range <= 0:  # Handle case of silent input
        print("  Warning: Input signal might be silent (dB range is zero or negative). Outputting silence.")
        vocals_audio = np.zeros_like(y)
        instruments_audio = np.zeros_like(y)
    else:
        if phase is None:
            print("Error: Phase information is missing for STFT reconstruction.")
            return
        # De-normalize using the min/max from the input
        pred_vocals_db = pred_vocals_norm * denormalization_range + min_db
        pred_instruments_db = pred_instruments_norm * denormalization_range + min_db
        # Convert back to amplitude
        pred_vocals_amp = librosa.db_to_amplitude(pred_vocals_db)
        pred_instruments_amp = librosa.db_to_amplitude(pred_instruments_db)
        # Combine with phase
        vocals_stft_rec = pred_vocals_amp * np.exp(1j * phase)
        instruments_stft_rec = pred_instruments_amp * np.exp(1j * phase)
        # Inverse STFT
        vocals_audio = librosa.istft(vocals_stft_rec, hop_length=HOP_LENGTH)
        instruments_audio = librosa.istft(instruments_stft_rec, hop_length=HOP_LENGTH)

    # --- Auto-suffix output filenames ---
    out_vocals_path = Path(output_vocals_wav)
    final_vocals_path = out_vocals_path.with_stem(f"{out_vocals_path.stem}_stft")

    out_instruments_path = Path(output_instruments_wav)
    final_instruments_path = out_instruments_path.with_stem(f"{out_instruments_path.stem}_stft")

    # Save outputs with new names
    if vocals_audio is not None and instruments_audio is not None:
        print(f"Saving outputs...")
        save_audio(vocals_audio, str(final_vocals_path))
        save_audio(instruments_audio, str(final_instruments_path))
        print(f"Saved vocals: {final_vocals_path}")
        print(f"Saved instruments: {final_instruments_path}")
    else:
        print("Skipping saving due to errors during audio reconstruction.")


def main():
    parser = argparse.ArgumentParser(
        description="Separate vocals and instruments from a mix WAV file using a trained U-Net with STFT spectrogram and min-max dB normalization consistent with converter_wav.py. Automatically appends _stft to output filenames."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained STFT model .pth file.",
    )
    parser.add_argument("--input_wav", type=str, required=True, help="Path to the input mix WAV file.")
    parser.add_argument(
        "--output_vocals",
        type=str,
        required=True,
        help="Base path to save the output vocals WAV file (e.g., output/song_vocals.wav). _stft will be added automatically.",
    )
    parser.add_argument(
        "--output_instruments",
        type=str,
        required=True,
        help="Base path to save the output instruments WAV file (e.g., output/song_instruments.wav). _stft will be added automatically.",
    )

    args = parser.parse_args()

    predict_wav(
        model_path=args.model,
        input_wav=args.input_wav,
        output_vocals_wav=args.output_vocals,
        output_instruments_wav=args.output_instruments,
    )


if __name__ == "__main__":
    main()
