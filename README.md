# Stem Separation - Vocal & Accompaniment

This project focuses on separating audio tracks into **vocal** and **accompaniment** stems using deep learning models (U-Nets). It includes scripts for data preparation, training (using Mel or STFT spectrograms), and prediction.

> 🧑‍🎓 This is the **final project** for the **Artificial Intelligence with Deep Learning** postgraduate course at **Universitat Politècnica de Catalunya (UPC)**.

## 🧠 Objective

Train a model capable of separating a mixed music track into:
- **Vocal track**
- **Accompaniment track** (everything else)

## 🛠️ Features

- **Two Spectrogram Options:** Train models using either Mel spectrograms or STFT spectrograms.
- **U-Net Architecture:** Utilizes a small U-Net model for the separation task.
- **Training Pipeline:** Includes data loading, training loop with validation, loss tracking, and model saving.
- **Prediction Script:** Allows separating vocals and instruments from a given WAV file using a trained model.
- **Sample Data:** Includes scripts to download and prepare sample audio data.

## 🔧 Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/aidl-2025-music-stem-separator.git # Replace with your repo URL if different
    cd aidl-2025-music-stem-separator
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ Usage Workflow

Follow these steps in order:

**1. Download Sample Data (Optional):**

   If you don't have your own audio data, you can download some sample tracks. This script will download them into the `sample_data/musdb/` directory.

   ```bash
   python sample_downloader/download.py
   ```

**2. Convert Audio to Spectrograms:**

   This script converts the raw audio files into Mel or STFT spectrograms (`.npy` files) and saves them to the specified output directory (default: `sample_data/musdb/spectrograms/`).

   *   **For Spectrograms:**
       ```bash
       python converter/convert.py 
       ```

**3. Train a Model:**

   Train a U-Net model using the generated spectrograms. Choose the type (`mel` or `stft`) and specify the directory containing the corresponding `.npy` files.

   *   **Train Mel Model:**
       ```bash
       python train.py --type mel --spectrogram_dir sample_data/spectrograms --epochs 50 --batch_size 8 --lr 0.001 --val_split 0.2
       ```
       *(Model saved to `u_net_mel/unet_small_mel.pth`, plot saved to `u_net_mel/unet_small_mel_loss_curve.png`)*

   *   **Train STFT Model:**
       ```bash
       python train.py --type stft --spectrogram_dir sample_data/spectrograms_stft --epochs 50 --batch_size 8 --lr 0.001 --val_split 0.2
       ```
       *(Model saved to `u_net_stft/unet_small_stft.pth`, plot saved to `u_net_stft/unet_small_stft_loss_curve.png`)*

   *Adjust `--epochs`, `--batch_size`, `--lr`, and `--val_split` as needed.*

**4. Predict (Separate Stems):**

   Use a trained model to separate a mix WAV file into vocals and instruments. Currently, only the Mel model prediction pipeline is fully implemented in `predict_wav.py`.

   ```bash
   python predict_wav.py \
       --model u_net_mel/unet_small_mel.pth \
       --input_wav eval/mixture.wav \
       --output_vocals output/vocals.wav \
       --output_instruments output/instruments.wav \
   ```
   *Make sure the output directories exist or adjust the paths.*
   *The `--is_mel` flag is required when using the Mel model.*

## 📦 Dataset (MUSDB18)

While sample data scripts are provided, this project is designed with the [MUSDB18 dataset](https://sigsep.github.io/datasets/musdb.html) in mind for more robust training.
- Download it manually if desired.
- You will need to adapt the `converter/convert.py` script or your workflow to process the MUSDB18 structure and place the generated spectrograms in a location accessible by `train.py`.

## ⏳ Status

- Core training and Mel prediction pipeline implemented.
- STFT training is available, but the `predict_wav.py` script needs adaptation for STFT models.