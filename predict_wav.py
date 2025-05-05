import argparse
import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

# Import both models
from u_net_mel.model import UNetSmall as MelUNet
from u_net_stft.model import UNetSmall as StftUNet

# Spectrogram parameters (MUST match converter/converter_wav.py)
N_MELS = 128
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
    sf.write(path, y, sr)

def predict_wav(model_path, input_wav, output_vocals_wav, output_instruments_wav, model_type='mel'):
    """Predict vocals and instruments from a WAV file using the trained model."""
    device = get_device()
    print(f"Using device: {device}")

    # Select model class based on type
    if model_type == 'mel':
        ModelClass = MelUNet
        print("Using MEL model")
    elif model_type == 'stft':
        ModelClass = StftUNet
        print("Using STFT model")
    else:
        raise ValueError("model_type must be 'mel' or 'stft'")

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

    # --- Calculate Input Spectrogram and Normalize (Consistent with converter_wav.py) ---
    min_db, max_db = None, None
    phase = None # Initialize phase

    if model_type == 'mel':
        print("Calculating MEL spectrogram...")
        mel_spec = librosa.feature.melspectrogram(y=y, sr=TARGET_SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        min_db = mel_db.min()
        max_db = mel_db.max()
        print(f"  Input MEL dB range: [{min_db:.2f}, {max_db:.2f}]")
        if max_db > min_db:
            spec_norm = (mel_db - min_db) / (max_db - min_db)
        else:
            spec_norm = np.zeros_like(mel_db)

    else: # stft
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
        pred_vocals_norm = model(spec_tensor).squeeze().cpu().numpy()

    # Ensure prediction is clipped to valid range [0, 1]
    pred_vocals_norm = np.clip(pred_vocals_norm, 0, 1)
    print(f"  Predicted vocals norm range: [{pred_vocals_norm.min():.2f}, {pred_vocals_norm.max():.2f}]")

    # Estimate instruments mask/spectrogram (as residual)
    pred_instruments_norm = spec_norm - pred_vocals_norm
    # Clip instrument estimate to ensure valid range
    pred_instruments_norm = np.clip(pred_instruments_norm, 0, 1)
    print(f"  Estimated instruments norm range: [{pred_instruments_norm.min():.2f}, {pred_instruments_norm.max():.2f}]")

    # --- Convert back to waveform using consistent de-normalization ---
    print("Converting predictions back to audio...")
    vocals_audio = None
    instruments_audio = None
    denormalization_range = max_db - min_db

    if denormalization_range <= 0: # Handle case of silent input
        print("  Warning: Input signal might be silent (dB range is zero or negative). Outputting silence.")
        vocals_audio = np.zeros_like(y)
        instruments_audio = np.zeros_like(y)
    elif model_type == 'mel':
        # De-normalize using the min/max from the input
        pred_vocals_db = pred_vocals_norm * denormalization_range + min_db
        pred_instruments_db = pred_instruments_norm * denormalization_range + min_db
        # Convert back to power
        pred_vocals_power = librosa.db_to_power(pred_vocals_db)
        pred_instruments_power = librosa.db_to_power(pred_instruments_db)
        # Inverse Mel
        vocals_audio = librosa.feature.inverse.mel_to_audio(
            pred_vocals_power, sr=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        instruments_audio = librosa.feature.inverse.mel_to_audio(
            pred_instruments_power, sr=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
    else: # stft
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
    final_vocals_path = out_vocals_path.with_stem(f"{out_vocals_path.stem}_{model_type}")

    out_instruments_path = Path(output_instruments_wav)
    final_instruments_path = out_instruments_path.with_stem(f"{out_instruments_path.stem}_{model_type}")

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
    parser = argparse.ArgumentParser(description="Separate vocals and instruments from a mix WAV file using a trained U-Net. Uses min-max dB normalization consistent with converter_wav.py. Automatically appends _mel or _stft to output filenames.") # Updated description

    parser.add_argument('--model', type=str, required=True, help="Path to the trained model .pth file.")
    parser.add_argument('--input_wav', type=str, required=True, help="Path to the input mix WAV file.")
    parser.add_argument('--output_vocals', type=str, required=True, help="Base path to save the output vocals WAV file (e.g., output/song_vocals.wav). _mel or _stft will be added automatically.")
    parser.add_argument('--output_instruments', type=str, required=True, help="Base path to save the output instruments WAV file (e.g., output/song_instruments.wav). _mel or _stft will be added automatically.")
    parser.add_argument('--type', type=str, default='mel', choices=['mel', 'stft'],
                        help="Type of model/spectrogram used for training ('mel' or 'stft'). Default: mel.") # Clarified help

    args = parser.parse_args()

    # Check if the correct model path is provided for the type (optional check)
    if args.type == 'mel' and 'stft' in args.model.lower():
        print("Warning: Model type is 'mel' but model path might contain 'stft'. Ensure correct model is loaded.")
    elif args.type == 'stft' and 'mel' in args.model.lower():
         print("Warning: Model type is 'stft' but model path might contain 'mel'. Ensure correct model is loaded.")

    predict_wav(
        model_path=args.model,
        input_wav=args.input_wav,
        output_vocals_wav=args.output_vocals,
        output_instruments_wav=args.output_instruments,
        model_type=args.type
    )

if __name__ == "__main__":
    main()