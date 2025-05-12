from u_net_stft.dataset import StftSpectrogramDataset
from u_net_stft.model import UNetSmall as StftUNet
from u_net_mel.dataset import ImprovedMelDataset
from u_net_mel.model import UNetSmall as MelUNet
import json
from pathlib import Path
import os
from datetime import datetime
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
import torch
import argparse
import warnings

# Suppress all UserWarnings containing GradScaler + CUDA
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*GradScaler is enabled, but CUDA is not available.*"
)


# Import your U-Net models and Dataset loaders

# --- Define paths ---
RESULTS_CSV_PATH = "training_results.csv"


def get_device():
    """
    Detect the best device available:
    - Use CUDA if available (NVIDIA GPU)
    - Use MPS if available (Apple Silicon GPU)
    - Otherwise fallback to CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def train_improved(
    model_class,
    dataset_class,
    model_save_dir,
    spectrogram_dir,
    num_epochs=100,
    batch_size=16,
    learning_rate=1e-4,
    val_split=0.2,
    chunk_size=128,
    overlap=0.5
):
    """
    Train a U-Net model on MEL or STFT spectrogram dataset with improved features.

    Args:
        model_class: U-Net model class to instantiate
        dataset_class: Dataset class to instantiate
        model_save_dir: Directory to save model checkpoints and training artifacts
        spectrogram_dir: Directory with processed spectrograms
        num_epochs: Number of training epochs
        batch_size: Number of samples per training batch
        learning_rate: Initial learning rate
        val_split: Fraction of data to use for validation
        chunk_size: Size of spectrogram chunks
        overlap: Overlap between chunks (0-1)
    """
    # Setup device
    device = get_device()
    print(f"Using device: {device}")

    # Create output directory
    model_save_dir = Path(model_save_dir)
    model_save_dir.mkdir(parents=True, exist_ok=True)

    # Create datasets
    full_dataset = dataset_class(
        spectrogram_dir,
        chunk_size=chunk_size,
        overlap=overlap,
        augment=True
    )
    train_size = int((1 - val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )

    print(f"Training set size: {train_size}")
    print(f"Validation set size: {val_size}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize model
    model = model_class(in_channels=1, out_channels=1).to(device)

    # Loss function and optimizer
    criterion = nn.L1Loss()  # L1 loss for better amplitude preservation
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Mixed precision training
    scaler = torch.amp.GradScaler(enabled=(device.type in ['cuda', 'mps']))
    autocast = torch.amp.autocast

    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    # Save training configuration
    config = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'val_split': val_split,
        'chunk_size': chunk_size,
        'overlap': overlap,
        'device': str(device),
        'model_type': model_class.__name__,
        'loss_function': 'L1Loss',
        'optimizer': 'AdamW',
        'scheduler': 'ReduceLROnPlateau'
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = model_save_dir / f'training_config_{timestamp}.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    print("Starting training...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for mix, vocals in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            mix, vocals = mix.to(device), vocals.to(device)

            optimizer.zero_grad()

            with autocast(device_type=device.type, enabled=(device.type in ['cuda', 'mps'])):
                pred = model(mix)
                loss = criterion(pred, vocals)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for mix, vocals in val_loader:
                mix, vocals = mix.to(device), vocals.to(device)
                pred = model(mix)
                loss = criterion(pred, vocals)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = model_save_dir / f'best_model_{timestamp}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, model_path)
            print(f"Saved best model with validation loss: {avg_val_loss:.6f}")

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.6f}")
        print(f"Val Loss: {avg_val_loss:.6f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

    # Save final model
    final_model_path = model_save_dir / f'final_model_{timestamp}.pth'
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
    }, final_model_path)

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(model_save_dir / f'training_curves_{timestamp}.png')
    plt.close()

    # Save results to CSV
    results_data = [
        timestamp,
        model_class.__name__,
        num_epochs,
        batch_size,
        learning_rate,
        val_split,
        avg_train_loss,
        avg_val_loss,
        str(final_model_path),
        str(model_save_dir / f'training_curves_{timestamp}.png')
    ]

    header = [
        'timestamp', 'model_type', 'num_epochs', 'batch_size', 'learning_rate',
        'val_split', 'final_train_loss', 'final_val_loss', 'model_save_path', 'plot_save_path'
    ]

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
    parser = argparse.ArgumentParser(description="Train improved U-Net model")
    parser.add_argument('--type', type=str, required=True, choices=['mel', 'stft'],
                        help="Choose which spectrogram type to train on (mel or stft)")
    parser.add_argument('--data_dir', type=str, required=True,
                        help="Directory containing processed spectrograms")
    parser.add_argument('--model_save_dir', type=str, required=True,
                        help="Directory to save trained models")
    parser.add_argument('--num_epochs', type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch size for training")
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help="Initial learning rate")
    parser.add_argument('--val_split', type=float, default=0.2,
                        help="Validation set size ratio")
    parser.add_argument('--chunk_size', type=int, default=128,
                        help="Size of spectrogram chunks")
    parser.add_argument('--overlap', type=float, default=0.5,
                        help="Overlap between chunks (0-1)")

    args = parser.parse_args()

    # Select correct model and dataset
    if args.type == 'mel':
        model_class = MelUNet
        dataset_class = ImprovedMelDataset
    else:
        model_class = StftUNet
        dataset_class = StftSpectrogramDataset

    train_improved(
        model_class=model_class,
        dataset_class=dataset_class,
        model_save_dir=args.model_save_dir,
        spectrogram_dir=args.data_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_split=args.val_split,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )


if __name__ == "__main__":
    main()
