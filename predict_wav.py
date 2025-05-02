import argparse
import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

# Import both models
from u_net_mel.model import UNetSmall as MelUNet
from u_net_stft.model import UNetSmall as StftUNet

def get_device():
    """Detect the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def load_audio(filepath, sr=44100):
    """Load a WAV file."""
    y, _ = librosa.load(filepath, sr=sr, mono=True)
    return y

def save_audio(y, path, sr=44100):
    """Save waveform to a WAV file."""
    sf.write(path, y, sr)

def mel_spectrogram(y, sr=44100, n_fft=2048, hop_length=512, n_mels=128):
    """Convert waveform to normalized mel spectrogram."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db + 80) / 80  # Normalize to [0,1]
    return mel_norm

def mel_to_audio(mel_norm, sr=44100, n_fft=2048, hop_length=512, n_mels=128):
    """Convert normalized mel spectrogram back to waveform."""
    mel_db = mel_norm * 80 - 80  # De-normalize
    mel_power = librosa.db_to_power(mel_db)
    y = librosa.feature.inverse.mel_to_audio(mel_power, sr=sr, n_fft=n_fft, hop_length=hop_length)
    return y

# --- Add STFT Functions ---
def stft_spectrogram(y, sr=44100, n_fft=2048, hop_length=512):
    """Convert waveform to normalized STFT magnitude spectrogram and return phase."""
    stft_result = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft_result)
    phase = np.angle(stft_result)

    # Normalize magnitude (e.g., dB scale like Mel, or simple min-max)
    # Using dB scale for consistency, range adjusted empirically
    mag_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    # Normalize to [0, 1] range. Adjust the range (-80, 0) if needed based on your data
    mag_norm = (mag_db + 80) / 80
    mag_norm = np.clip(mag_norm, 0, 1) # Ensure values are within [0, 1]

    return mag_norm, phase

def stft_to_audio(mag_norm, phase, sr=44100, n_fft=2048, hop_length=512):
    """Convert normalized STFT magnitude spectrogram and phase back to waveform."""
    # De-normalize magnitude
    mag_db = mag_norm * 80 - 80
    magnitude = librosa.db_to_amplitude(mag_db)

    # Combine magnitude and phase
    stft_reconstructed = magnitude * np.exp(1j * phase)

    # Inverse STFT
    y = librosa.istft(stft_reconstructed, hop_length=hop_length)
    return y

def predict_wav(model_path, input_wav, output_vocals_wav, output_instruments_wav, model_type='mel'):
    """Predict vocals and instruments from a WAV file using the trained model."""
    device = get_device()
    print(f"Using device: {device}")

    # Select model class based on type
    if model_type == 'mel':
        ModelClass = MelUNet
    elif model_type == 'stft':
        ModelClass = StftUNet
    else:
        raise ValueError("model_type must be 'mel' or 'stft'")

    # Load trained model
    model = ModelClass(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load input audio
    y = load_audio(input_wav)

    # Compute spectrogram (and phase for STFT)
    if model_type == 'mel':
        spec_norm = mel_spectrogram(y)
        phase = None # Not needed for Mel reconstruction here
    else: # stft
        spec_norm, phase = stft_spectrogram(y)

    # Prepare input for model
    spec_tensor = torch.tensor(spec_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Predict the vocals mask/spectrogram
    with torch.no_grad():
        pred_vocals_norm = model(spec_tensor).squeeze().cpu().numpy()

    # Ensure prediction is clipped to valid range [0, 1]
    pred_vocals_norm = np.clip(pred_vocals_norm, 0, 1)

    # Estimate instruments mask/spectrogram (as residual)
    # Note: This assumes the model predicts the vocal spectrogram directly.
    # If it predicts a mask, the logic would be different (e.g., pred_instruments = spec_norm * (1 - mask))
    pred_instruments_norm = spec_norm - pred_vocals_norm
    # Clip instrument estimate to ensure valid range
    pred_instruments_norm = np.clip(pred_instruments_norm, 0, 1)


    # Convert back to waveform
    if model_type == 'mel':
        vocals_audio = mel_to_audio(pred_vocals_norm)
        instruments_audio = mel_to_audio(pred_instruments_norm)
    else: # stft - requires phase
        vocals_audio = stft_to_audio(pred_vocals_norm, phase)
        instruments_audio = stft_to_audio(pred_instruments_norm, phase)


    # --- Auto-suffix output filenames ---
    out_vocals_path = Path(output_vocals_wav)
    final_vocals_path = out_vocals_path.with_stem(f"{out_vocals_path.stem}_{model_type}")

    out_instruments_path = Path(output_instruments_wav)
    final_instruments_path = out_instruments_path.with_stem(f"{out_instruments_path.stem}_{model_type}")

    # Save outputs with new names
    save_audio(vocals_audio, str(final_vocals_path))
    save_audio(instruments_audio, str(final_instruments_path))
    print(f"Saved vocals: {final_vocals_path}")
    print(f"Saved instruments: {final_instruments_path}")

def main():
    parser = argparse.ArgumentParser(description="Separate vocals and instruments from a mix WAV file using a trained U-Net. Automatically appends _mel or _stft to output filenames.")

    parser.add_argument('--model', type=str, required=True, help="Path to the trained model .pth file.")
    parser.add_argument('--input_wav', type=str, required=True, help="Path to the input mix WAV file.")
    parser.add_argument('--output_vocals', type=str, required=True, help="Base path to save the output vocals WAV file (e.g., output/song_vocals.wav). _mel or _stft will be added automatically.")
    parser.add_argument('--output_instruments', type=str, required=True, help="Base path to save the output instruments WAV file (e.g., output/song_instruments.wav). _mel or _stft will be added automatically.")
    parser.add_argument('--type', type=str, default='mel', choices=['mel', 'stft'],
                        help="Type of model/spectrogram to use ('mel' or 'stft'). Default: mel.")

    args = parser.parse_args()

    # Check if the correct model path is provided for the type
    if args.type == 'mel' and 'stft' in args.model.lower():
        print("Warning: Model type is 'mel' but model path might contain 'stft'.")
    elif args.type == 'stft' and 'mel' in args.model.lower():
         print("Warning: Model type is 'stft' but model path might contain 'mel'.")

    predict_wav(
        model_path=args.model,
        input_wav=args.input_wav,
        output_vocals_wav=args.output_vocals,
        output_instruments_wav=args.output_instruments,
        model_type=args.type # Pass the type
    )

if __name__ == "__main__":
    main()