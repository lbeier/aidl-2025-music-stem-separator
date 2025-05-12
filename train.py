import warnings
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
from datetime import datetime
import os
import shutil
import numpy as np
import librosa
import librosa.display
import soundfile as sf
from pathlib import Path
import re
import math

from u_net_mel.medium import UNetMedium
from u_net_mel.model import UNetSmall
from u_net_mel.dataset import MelSpectrogramDataset
from u_net_stft.model import UNetSmall as StftUNet
from u_net_stft.dataset import StftSpectrogramDataset

RESULTS_CSV_PATH = "training_results.csv"

warnings.filterwarnings("ignore", category=UserWarning, message=".*GradScaler is enabled, but CUDA is not available.*")

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def train(model_class, dataset_class, model_save_path, spectrogram_dir, num_epochs=50, batch_size=8, lr=1e-3, val_split=0.2, spec_type='mel'):
    device = get_device()
    print(f"Using device: {device}")

    save_output_dir = os.path.join(os.path.dirname(__file__), "interpretability")
    if os.path.exists(save_output_dir):
        shutil.rmtree(save_output_dir)
    os.makedirs(save_output_dir, exist_ok=True)

    full_dataset = dataset_class(spectrogram_dir)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Training set size: {train_size}")
    print(f"Validation set size: {val_size}")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    print("DataLoader initialized, starting training loop...")

    model = model_class(in_channels=1, out_channels=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.L1Loss()

    scaler = torch.amp.GradScaler(enabled=(device.type in ['cuda', 'mps']))
    autocast = torch.amp.autocast

    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        losses = []
        total_batches = math.floor(len(train_dataset) / batch_size)
        pred_vocals_all = []
        gt_vocals_all = []
        valid_examples = 0
        example_song_name = None
        epoch_dir = os.path.join(save_output_dir, f"epoch_{epoch+1:02d}")
        os.makedirs(epoch_dir, exist_ok=True)

        for batch_idx, (mix, vocals, instruments) in enumerate(train_dataloader):
            if batch_idx >= total_batches:
                break

            mix, vocals, instruments = mix.to(device), vocals.to(device), instruments.to(device)

            optimizer.zero_grad()
            with autocast(device_type=device.type, enabled=(device.type in ['cuda', 'mps'])):
                output = model(mix)
                pred_vocals = output[:, 0:1, :, :]
                pred_instruments = output[:, 1:2, :, :]
                loss = criterion(pred_vocals, vocals) + criterion(pred_instruments, instruments)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            losses.append(loss.item())

            if batch_idx % 100 == 0:
                avg_loss = sum(losses[-100:]) / len(losses[-100:])
                print(f"Epoch {epoch+1} | Batch {batch_idx+1}/{total_batches} | Avg Loss (last 100): {avg_loss:.4f}")

            vocal_np = vocals[0].detach().cpu().numpy()
            if valid_examples < 30 and np.any(vocal_np > 0.01):
                pred_vocals_all.append(pred_vocals[0].detach().cpu().numpy())
                gt_vocals_all.append(vocal_np)
                if example_song_name is None:
                    mix_path = full_dataset.chunk_files[batch_idx]
                    match = re.search(r"chunk_(\d+)", mix_path.stem)
                    chunk_id = match.group(1) if match else "000"
                    example_song_name = str(mix_path.name).replace(f"_mix_mel_chunk_{chunk_id}.npy", "").replace(f"_mix_stft_chunk_{chunk_id}.npy", "")
                    example_song_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', example_song_name)
                valid_examples += 1

        pred_concat = np.hstack(pred_vocals_all)
        gt_concat = np.hstack(gt_vocals_all)

        def to_audio(mel):
            mel_db = mel * 80 - 80
            mel_db = mel_db.astype(np.float32)
            audio = librosa.feature.inverse.mel_to_audio(mel_db, sr=44100, n_fft=2048, hop_length=512)
            if (
                np.isnan(audio).any()
                or np.isinf(audio).any()
                or audio.shape[0] == 0
                or not np.any(np.abs(audio) > 1e-5)
            ):
                raise ValueError("Invalid or silent audio — skipping save")
            return audio

        if len(pred_concat) > 0:
            try:
                pred_audio = to_audio(pred_concat)
                sf.write(f"{epoch_dir}/{example_song_name}_pred_vocals.wav", pred_audio, 44100)
            except Exception as e:
                print(f"Skipped writing pred_vocals.wav: {e}")

            try:
                gt_audio = to_audio(gt_concat)
                sf.write(f"{epoch_dir}/{example_song_name}_gt_vocals.wav", gt_audio, 44100)
            except Exception as e:
                print(f"Skipped writing gt_vocals.wav: {e}")

            try:
                fig, ax = plt.subplots(figsize=(12, 4))
                spec_db = (pred_concat * 80 - 80).astype(np.float32)
                if spec_db.ndim == 3:
                    spec_db = spec_db.squeeze(0)
                if spec_db.ndim != 2:
                    raise ValueError(f"Spectrogram has invalid shape: {spec_db.shape}")
                librosa.display.specshow(spec_db, y_axis='mel', x_axis='time', cmap='magma', vmin=-80, vmax=0)
                plt.title("Predicted Vocals - Full Example")
                plt.colorbar(format='%+2.0f dB')
                plt.tight_layout()
                fig.savefig(f"{epoch_dir}/{example_song_name}_pred_vocals.png")
                plt.close(fig)
            except Exception as e:
                print(f"Failed to save spectrogram: {e}")

        avg_train_loss = running_loss / total_batches
        train_losses.append(avg_train_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.6f}")

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (L1)')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plot_save_path = model_save_path.replace(".pth", "_loss_curve.png")
    plt.savefig(plot_save_path)
    print(f"Loss curve saved at {plot_save_path}")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_type = 'mel' if 'mel' in model_save_path else 'stft' if 'stft' in model_save_path else 'unknown'
    results_data = [timestamp, model_type, num_epochs, batch_size, lr, val_split, train_losses[-1], 'N/A', model_save_path, plot_save_path]
    header = ['timestamp', 'model_type', 'num_epochs', 'batch_size', 'learning_rate', 'val_split', 'final_train_loss', 'final_val_loss', 'model_save_path', 'plot_save_path']
    file_exists = os.path.exists(RESULTS_CSV_PATH)
    try:
        with open(RESULTS_CSV_PATH, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(results_data)
        print(f"Results appended to {RESULTS_CSV_PATH}")
    except IOError as e:
        print(f"Error writing to CSV {RESULTS_CSV_PATH}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Train U-Net on MEL or STFT spectrograms.")
    parser.add_argument('--type', type=str, required=True, choices=['mel', 'stft'], help="Choose which spectrogram type to train on (mel or stft).")
    parser.add_argument('--spectrogram_dir', type=str, default="sample_data/spectrograms", help="Directory where spectrogram .npy files are stored.")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size during training.")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for optimizer.")
    parser.add_argument('--val_split', type=float, default=0.2, help="Fraction of data for validation set (default: 0.2).")

    args = parser.parse_args()

    if args.type == 'mel':
        model_class = UNetMedium
        dataset_class = MelSpectrogramDataset
        model_save_path = "u_net_mel/unet_medium_mel.pth"
        spec_type = 'mel'
    else:
        model_class = StftUNet
        dataset_class = StftSpectrogramDataset
        model_save_path = "u_net_stft/unet_small_stft.pth"
        spec_type = 'stft'

    train(
        model_class=model_class,
        dataset_class=dataset_class,
        model_save_path=model_save_path,
        spectrogram_dir=args.spectrogram_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        spec_type=spec_type
    )

if __name__ == "__main__":
    main()
