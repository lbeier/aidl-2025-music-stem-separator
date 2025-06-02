import argparse
import csv  # Import csv module
import logging
import os  # Import os for checking file existence
import warnings
from datetime import datetime  # Import datetime for timestamps

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from u_net_stft.dataset import StftSpectrogramDataset
from u_net_stft.model import UNetSmall as StftUNet
from u_net_stft.h5_dataset import H5SpectrogramDataset
from u_net_stft.augment import spec_augment

# Suppress all UserWarnings containing GradScaler + CUDA
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*GradScaler is enabled, but CUDA is not available.*",
)


LOG_FILENAME = "epochs.log"
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)

# --- Define CSV file path ---
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


def train(
    model_class,
    dataset_class,
    model_save_path,
    dataset_args,
    num_epochs=50,
    batch_size=8,
    lr=1e-3,
    val_split=0.2,
    resume_from=None,
):
    """
    Train a U-Net model on STFT spectrogram dataset.

    Args:
        model_class: U-Net model class to instantiate.
        dataset_class: Dataset class to instantiate.
        model_save_path: File path to save the trained model.
        dataset_args: dict of arguments for the dataset (e.g. h5_path, transform)
        num_epochs: Number of training epochs.
        batch_size: Number of samples per training batch.
        lr: Learning rate.
        val_split: Fraction of data to use for validation (e.g., 0.2 for 20%).
    Targets:
        Both vocals and instruments are used as targets for training.
    """
    # Select the best available device
    device = get_device()
    device_message = f"Using device: {device}"
    print(device_message)
    logging.debug(device_message)

    # Load full dataset
    full_dataset = dataset_class(**dataset_args)

    # Split dataset into training and validation
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size])

    set_size = f"Training set size: {train_size}"
    validation_size = f"Validation set size: {val_size}"
    print(set_size)
    print(validation_size)
    logging.debug(set_size)
    logging.debug(validation_size)

    # Dataloaders for batching
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,  # You can increase if you have good CPU
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation data
        num_workers=6,
        pin_memory=True,
    )

    # Instantiate model and move it to the device
    model = model_class(in_channels=1, out_channels=1).to(device)

    if resume_from is not None:
        checkpoint_resume = f"Resuming from checkpoint: {resume_from}"
        print(checkpoint_resume)
        logging.debug(checkpoint_resume)
        model.load_state_dict(torch.load(resume_from, map_location=device))

    # Use Mean Absolute Error loss (L1 Loss)
    criterion = nn.L1Loss()

    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Setup automatic mixed precision:
    scaler = torch.amp.GradScaler(enabled=(device.type in ["cuda", "mps"]))
    autocast = torch.amp.autocast

    # Track losses
    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_ckpt_path = ""

    # --- Start Training ---
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        # --- Training Phase ---
        model.train()  # Set model to training mode
        running_loss = 0.0
        # Loop through all training batches with tqdm
        for mix, vocals, instruments in tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs} - Train",
            leave=False,
        ):
            mix, vocals, instruments = mix.to(device), vocals.to(
                device), instruments.to(device)

            optimizer.zero_grad()

            # Mixed precision forward and loss computation
            with autocast(device_type=device.type, enabled=(device.type in ["cuda", "mps"])):
                outputs = model(mix)
                vocal_mask = outputs[:, 0:1]
                instr_mask = outputs[:, 1:2]
                pred_vocals = vocal_mask * mix
                pred_instr = instr_mask * mix
                loss_vocals = criterion(pred_vocals, vocals)
                loss_instruments = criterion(pred_instr, instruments)
                loss = loss_vocals + loss_instruments

            # Backward pass with scaled loss
            scaler.scale(loss).backward()

            # Optimizer step
            scaler.step(optimizer)

            # Update the scaling factor
            scaler.update()

            # Accumulate total loss for this epoch
            running_loss += loss.item()

        # Average training loss across all batches
        avg_train_loss = running_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # --- Validation Phase ---
        model.eval()  # Set model to evaluation mode
        running_val_loss = 0.0
        with torch.no_grad():  # Disable gradient calculation
            # Loop through all validation batches with tqdm
            for mix, vocals, instruments in tqdm(
                val_dataloader,
                desc=f"Epoch {epoch + 1}/{num_epochs} - Val",
                leave=False,
            ):
                mix, vocals, instruments = mix.to(device), vocals.to(
                    device), instruments.to(device)

                # Use autocast only if enabled, but don't compute gradients
                with autocast(device_type=device.type, enabled=(device.type in ["cuda", "mps"])):
                    outputs = model(mix)
                    vocal_mask = outputs[:, 0:1]
                    instr_mask = outputs[:, 1:2]
                    pred_vocals = vocal_mask * mix
                    pred_instr = instr_mask * mix
                    loss_vocals = criterion(pred_vocals, vocals)
                    loss_instruments = criterion(pred_instr, instruments)
                    loss = loss_vocals + loss_instruments

                running_val_loss += loss.item()

        # Average validation loss across all batches
        avg_val_loss = running_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)

        message = f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"

        print(message)
        logging.debug(message)

        # Save checkpoint for this epoch
        epoch_ckpt_path = model_save_path.replace(
            ".pth", f"_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), epoch_ckpt_path)
        checkopoint_saved = f"Checkpoint saved at {epoch_ckpt_path}"
        print(checkopoint_saved)
        logging.debug(checkopoint_saved)

        # Save best model checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_ckpt_path = model_save_path.replace(".pth", "_best.pth")
            torch.save(model.state_dict(), best_ckpt_path)

    # --- End Training ---

    # Save the trained model weights
    torch.save(model.state_dict(), model_save_path)
    model_saved = f"Model saved at {model_save_path}"
    print(model_saved)
    logging.debug(model_saved)

    # --- Plotting ---
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plot_save_path = model_save_path.replace(".pth", "_loss_curve.png")
    plt.savefig(plot_save_path)
    loss_curve_saved = f"Loss curve saved at {plot_save_path}"
    print(loss_curve_saved)
    logging.debug(loss_curve_saved)
    # Optionally display the plot
    # plt.show()

    # --- Save results to CSV ---
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_type = "stft"
    final_train_loss = train_losses[-1] if train_losses else float("nan")
    final_val_loss = val_losses[-1] if val_losses else float("nan")

    results_data = [
        timestamp,
        model_type,
        num_epochs,
        batch_size,
        lr,
        val_split,
        final_train_loss,
        final_val_loss,
        model_save_path,
        plot_save_path,
        best_val_loss,
        best_ckpt_path if best_val_loss < float("inf") else "",
    ]

    header = [
        "timestamp",
        "model_type",
        "num_epochs",
        "batch_size",
        "learning_rate",
        "val_split",
        "final_train_loss",
        "final_val_loss",
        "model_save_path",
        "plot_save_path",
        "best_val_loss",
        "best_model_path",
    ]

    file_exists = os.path.exists(RESULTS_CSV_PATH)

    try:
        with open(RESULTS_CSV_PATH, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(header)  # Write header only if file is new
            writer.writerow(results_data)
        results_appended = f"Results appended to {RESULTS_CSV_PATH}"
        print(results_appended)
        logging.debug(results_appended)
    except IOError as e:
        print(f"Error writing to CSV {RESULTS_CSV_PATH}: {e}")


def main():
    """
    Main CLI entry point.
    Parses command line arguments and starts training.
    """
    parser = argparse.ArgumentParser(
        description="Train U-Net on STFT spectrograms or .h5 spectrograms.")

    parser.add_argument(
        "--spectrogram_dir",
        type=str,
        default=None,
        help="Directory where spectrogram .npy files are stored (for legacy .npy pipeline).",
    )
    parser.add_argument(
        "--h5_path",
        type=str,
        default=None,
        help="Path to .h5 file with spectrograms (recommended).",
    )
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size during training.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for optimizer.")
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Fraction of data for validation set (default: 0.2).",
    )
    parser.add_argument(
        "--resume_from", type=str, default=None, help="Optional path to a checkpoint .pth file to resume training from."
    )

    args = parser.parse_args()

    model_class = StftUNet
    # Decide dataset
    if args.h5_path is not None:
        dataset_class = H5SpectrogramDataset
        dataset_args = {"h5_path": args.h5_path, "transform": spec_augment}
        print(f"Using H5SpectrogramDataset with {args.h5_path}")
    elif args.spectrogram_dir is not None:
        dataset_class = StftSpectrogramDataset
        dataset_args = {"spectrogram_dir": args.spectrogram_dir}
        print(f"Using StftSpectrogramDataset with {args.spectrogram_dir}")
    else:
        raise ValueError(
            "You must provide either --h5_path or --spectrogram_dir")

    model_save_path = StftUNet.MODEL_SAVE_PATH

    # Start training
    train(
        model_class=model_class,
        dataset_class=dataset_class,
        model_save_path=model_save_path,
        dataset_args=dataset_args,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
