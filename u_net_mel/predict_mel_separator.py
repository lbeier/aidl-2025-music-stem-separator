import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from model import UNetSmall  # Assumes your UNetSmall is accessible
from u_net_mel.medium import UNetMedium

# --- Configurable params ---
SR = 44100
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
CHUNK_SIZE = 128

def preprocess_wav_to_mel_chunks(wav_path: Path) -> list:
    y, sr = librosa.load(wav_path, sr=SR, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = np.clip((mel_db + 80) / 80, 0, 1).astype(np.float32)

    chunks = []
    num_chunks = mel_norm.shape[1] // CHUNK_SIZE
    for i in range(num_chunks):
        chunk = mel_norm[:, i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]
        chunks.append(torch.tensor(chunk).unsqueeze(0).unsqueeze(0))  # [1, 1, 128, T]
    return chunks

def predict(model: torch.nn.Module, chunks: list, device) -> tuple:
    model.eval()
    vocals, instruments = [], []
    with torch.no_grad():
        for chunk in chunks:
            chunk = chunk.to(device)
            out = model(chunk)
            vocals.append(out[:, 0].squeeze(0).cpu().numpy())
            instruments.append(out[:, 1].squeeze(0).cpu().numpy())
    return np.hstack(vocals), np.hstack(instruments)

def mel_to_audio(mel_norm: np.ndarray) -> np.ndarray:
    mel_db = mel_norm * 80 - 80
    mel_power = librosa.db_to_power(mel_db)
    return librosa.feature.inverse.mel_to_audio(mel_power, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH)

def separate_and_save(wav_path: str, model_weights: str, out_dir: str):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # model = UNetSmall(in_channels=1, out_channels=2).to(device)
    model = UNetMedium(in_channels=1, out_channels=2, base_channels=64).to(device)
    model.load_state_dict(torch.load(model_weights, map_location=device))
    model.eval()

    chunks = preprocess_wav_to_mel_chunks(Path(wav_path))
    pred_vocals, pred_instr = predict(model, chunks, device)

    wav_vocals = mel_to_audio(pred_vocals)
    wav_instr = mel_to_audio(pred_instr)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    sf.write(out_path / "vocals.wav", wav_vocals, SR)
    sf.write(out_path / "instruments.wav", wav_instr, SR)
    print("Saved vocals.wav and instruments.wav")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True, help="Path to input WAV file")
    parser.add_argument("--model", required=True, help="Path to trained .pth model")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    separate_and_save(args.wav, args.model, args.out)
