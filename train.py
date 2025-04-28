import warnings

# Suppress all UserWarnings containing GradScaler + CUDA
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*GradScaler is enabled, but CUDA is not available.*"
)

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import your U-Net models and Dataset loaders
from u_net_mel.model import UNetSmall as MelUNet
from u_net_mel.dataset import MelSpectrogramDataset
from u_net_stft.model import UNetSmall as StftUNet
from u_net_stft.dataset import StftSpectrogramDataset

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

def train(model_class, dataset_class, model_save_path, spectrogram_dir, num_epochs=50, batch_size=8, lr=1e-3):
    """
    Train a U-Net model on MEL or STFT spectrogram dataset.

    Args:
        model_class: U-Net model class to instantiate.
        dataset_class: Dataset class to instantiate.
        model_save_path: File path to save the trained model.
        spectrogram_dir: Directory with .npy spectrograms.
        num_epochs: Number of training epochs.
        batch_size: Number of samples per training batch.
        lr: Learning rate.
    """
    # Select the best available device
    device = get_device()
    print(f"Using device: {device}")

    # Load dataset
    dataset = dataset_class(spectrogram_dir)

    # Dataloader for batching
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2  # You can increase if you have good CPU
    )

    # Instantiate model and move it to the device
    model = model_class(in_channels=1, out_channels=1).to(device)

    # Mean Squared Error loss (good for spectrogram pixel prediction)
    criterion = nn.MSELoss()

    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Setup automatic mixed precision:
    scaler = torch.amp.GradScaler(enabled=(device.type in ['cuda', 'mps']))
    autocast = torch.amp.autocast

    # Set model in training mode
    model.train()

    # --- Start Training ---
    for epoch in range(num_epochs):
        running_loss = 0.0

        # Loop through all batches
        for mix, vocals in dataloader:
            mix, vocals = mix.to(device), vocals.to(device)

            optimizer.zero_grad()

            # Mixed precision forward and loss computation
            with autocast(device_type=device.type, enabled=(device.type in ['cuda', 'mps'])):
                outputs = model(mix)
                loss = criterion(outputs, vocals)

            # Backward pass with scaled loss
            scaler.scale(loss).backward()

            # Optimizer step
            scaler.step(optimizer)

            # Update the scaling factor
            scaler.update()

            # Accumulate total loss for this epoch
            running_loss += loss.item()

        # Average loss across all batches
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.6f}")

    # Save the trained model weights
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")

def main():
    """
    Main CLI entry point.
    Parses command line arguments and starts training.
    """
    parser = argparse.ArgumentParser(description="Train U-Net on MEL or STFT spectrograms.")

    parser.add_argument('--type', type=str, required=True, choices=['mel', 'stft'],
                        help="Choose which spectrogram type to train on (mel or stft).")
    parser.add_argument('--spectrogram_dir', type=str, default="sample_data/spectrograms",
                        help="Directory where spectrogram .npy files are stored.")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size during training.")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for optimizer.")

    args = parser.parse_args()

    # Select correct model and dataset
    if args.type == 'mel':
        model_class = MelUNet
        dataset_class = MelSpectrogramDataset
        model_save_path = "u_net_mel/unet_small_mel.pth"
    else:
        model_class = StftUNet
        dataset_class = StftSpectrogramDataset
        model_save_path = "u_net_stft/unet_small_stft.pth"

    # Start training
    train(
        model_class=model_class,
        dataset_class=dataset_class,
        model_save_path=model_save_path,
        spectrogram_dir=args.spectrogram_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )

if __name__ == "__main__":
    main()