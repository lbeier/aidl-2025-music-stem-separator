import argparse
import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

from u_net_mel.model import UNetSmall  # Change to u_net_stft if you want STFT later

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

def predict_wav(model_path, input_wav, output_vocals_wav, output_instruments_wav, is_mel=True):
    """Predict vocals and instruments from a WAV file using the trained model."""
    device = get_device()
    print(f"Using device: {device}")

    # Load trained model
    model = UNetSmall(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load input audio
    y = load_audio(input_wav)

    # Compute spectrogram
    if is_mel:
        spec = mel_spectrogram(y)
    else:
        raise NotImplementedError("Currently only MEL model is supported.")

    # Prepare input for model
    spec_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        pred_vocals = model(spec_tensor).squeeze().cpu().numpy()

    # Estimate instruments (residual)
    pred_instruments = spec - pred_vocals
    pred_instruments = np.clip(pred_instruments, 0, 1)

    # Convert back to waveform
    vocals_audio = mel_to_audio(pred_vocals)
    instruments_audio = mel_to_audio(pred_instruments)

    # Save outputs
    save_audio(vocals_audio, output_vocals_wav)
    save_audio(instruments_audio, output_instruments_wav)
    print(f"Saved vocals: {output_vocals_wav}")
    print(f"Saved instruments: {output_instruments_wav}")

def main():
    parser = argparse.ArgumentParser(description="Separate vocals and instruments from a mix WAV file using a trained U-Net.")

    parser.add_argument('--model', type=str, required=True, help="Path to the trained model .pth file.")
    parser.add_argument('--input_wav', type=str, required=True, help="Path to the input mix WAV file.")
    parser.add_argument('--output_vocals', type=str, required=True, help="Path to save the output vocals WAV file.")
    parser.add_argument('--output_instruments', type=str, required=True, help="Path to save the output instruments WAV file.")
    parser.add_argument('--is_mel', action='store_true', help="Use MEL spectrogram model (default).")

    args = parser.parse_args()

    predict_wav(
        model_path=args.model,
        input_wav=args.input_wav,
        output_vocals_wav=args.output_vocals,
        output_instruments_wav=args.output_instruments,
        is_mel=args.is_mel
    )

if __name__ == "__main__":
    main()