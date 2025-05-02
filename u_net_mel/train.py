import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import model and dataset
from model import UNetSmall
from dataset import MelSpectrogramDataset

def train():
    """
    Train the U-Net model on MEL spectrogram data.
    """

    # --- Configuration ---
    spectrogram_dir = "sample_data/spectrograms"  # Path to directory with .npy spectrograms

    lr = 1e-3              # Learning rate for optimizer
    batch_size = 8         # Number of samples per training batch
    num_epochs = 50        # Total number of epochs to train

    # Select best device: MPS (Mac GPU) if available, then CUDA, then CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Dataset and Dataloader ---
    dataset = MelSpectrogramDataset(spectrogram_dir)  # Load the custom dataset

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,   # Shuffle data each epoch for better generalization
        num_workers=2   # Number of parallel CPU workers for data loading
    )

    # --- Model, Loss, Optimizer ---
    model = UNetSmall(in_channels=1, out_channels=1).to(device)  # Instantiate U-Net

    criterion = nn.MSELoss()  # Loss function: Mean Squared Error (good for spectrogram pixel regression)

    optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam optimizer

    # --- Set model to training mode ---
    model.train()

    # --- Training Loop ---
    for epoch in range(num_epochs):
        loss_sum = 0.0  # Accumulate total loss over the epoch

        for mix, vocals in loader:
            # Move data to device
            mix, vocals = mix.to(device), vocals.to(device)

            # Forward pass: Predict vocals from mix
            preds = model(mix)

            # Compute loss between prediction and ground-truth vocals
            loss = criterion(preds, vocals)

            # Backward pass: Clear previous gradients
            optimizer.zero_grad()

            # Compute new gradients
            loss.backward()

            # Update model weights
            optimizer.step()

            # Accumulate batch loss
            loss_sum += loss.item()

        # Average loss across all batches
        avg_loss = loss_sum / len(loader)

        # Print epoch loss
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.6f}")

    # --- Save Trained Model ---
    torch.save(model.state_dict(), "unet_small_mel.pth")
    print("Model saved to unet_small_mel.pth")

if __name__ == "__main__":
    # If run as script, start training
    train()