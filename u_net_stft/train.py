import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import U-Net model and STFT dataset
from model import UNetSmall
from dataset import StftSpectrogramDataset

def train():
    """
    Train U-Net model on STFT spectrogram data.
    """

    # --- Configuration ---
    spectrogram_dir = "../../sample_data/spectrograms"  # Path to the directory with STFT .npy files

    lr = 1e-3            # Learning rate for Adam optimizer
    batch_size = 8       # Number of samples per batch
    num_epochs = 50      # Number of epochs to train

    # Select best available device (CUDA if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Dataset and DataLoader ---
    dataset = StftSpectrogramDataset(spectrogram_dir)  # Instantiate dataset

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,     # Shuffle dataset every epoch
        num_workers=2     # Number of subprocesses for data loading (can be increased)
    )

    # --- Model, Loss Function, Optimizer ---
    model = UNetSmall(in_channels=1, out_channels=1).to(device)  # U-Net model for single-channel spectrogram input/output

    criterion = nn.MSELoss()  # Mean Squared Error Loss (good for spectrogram pixel prediction)

    optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam optimizer for weight updates

    # --- Set model to training mode ---
    model.train()

    # --- Training Loop ---
    for epoch in range(num_epochs):
        loss_sum = 0.0  # Accumulate loss over the epoch

        for mix, vocals in loader:
            # Move batch to device
            mix, vocals = mix.to(device), vocals.to(device)

            # Forward pass: predict vocals from mix
            preds = model(mix)

            # Compute loss
            loss = criterion(preds, vocals)

            # Backward pass
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()        # Backpropagate gradients
            optimizer.step()       # Update model parameters

            # Accumulate batch loss
            loss_sum += loss.item()

        # Compute average loss over all batches
        avg_loss = loss_sum / len(loader)

        # Print epoch loss
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.6f}")

    # --- Save trained model ---
    torch.save(model.state_dict(), "unet_small_stft.pth")
    print("Model saved to unet_small_stft.pth")

if __name__ == "__main__":
    # Entry point when run as a script
    train()