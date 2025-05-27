# Stem Separation - Vocal & Accompaniment

This project focuses on separating audio tracks into **vocal** and **accompaniment** stems using deep learning models (U-Nets). It includes scripts for data preparation, training (using STFT spectrograms), and prediction.

> 🧑‍🎓 This is the **final project** for the **Artificial Intelligence with Deep Learning** postgraduate course at **Universitat Politècnica de Catalunya (UPC)**.

## 🧠 Objective

Train a model capable of separating a mixed music track into:

- **Vocal track**
- **Accompaniment track** (everything else)

## 🛠️ Features

- **Spectrograms:** Train models using STFT spectrograms.
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

This script converts the raw audio files into STFT spectrograms (`.npy` files) and saves them to the specified output directory (default: `sample_data/musdb/spectrograms/`).

```bash
python converter/convert.py
```

**3. Train a Model:**

Train a U-Net model using the generated spectrograms. Choose the type (`stft`) and specify the directory containing the corresponding `.npy` files.

- **Train STFT Model:**
  ```bash
  python train.py --type stft --spectrogram_dir sample_data/spectrograms_stft --epochs 50 --batch_size 8 --lr 0.001 --val_split 0.2
  ```
  _(Model saved to `u_net_stft/unet_small_stft.pth`, plot saved to `u_net_stft/unet_small_stft_loss_curve.png`)_

_Adjust `--epochs`, `--batch_size`, `--lr`, and `--val_split` as needed._

**4. Predict (Separate Stems):**

Use the prediction script to separate vocals and instruments from a mix WAV file using a trained model. Example:

```bash
python predict_wav.py \
    --model u_net_stft/unet_small_stft.pth \
    --input_wav path/to/mix.wav \
    --output_vocals output/pred_vocals.wav \
    --output_instruments output/pred_instruments.wav
```

This will generate `output/pred_vocals_stft.wav` and `output/pred_instruments_stft.wav`.

**5. Analyze Separation Results:**

Use the analysis script to evaluate the quality of the separation. You can provide the original mix, the predicted stems, and (optionally) reference stems for SDR and detailed analysis:

```bash
python analyze_separation.py \
    --mix path/to/mix.wav \
    --vocals output/pred_vocals_stft.wav \
    --instruments output/pred_instruments_stft.wav \
    --ref_vocals path/to/reference_vocals.wav \
    --ref_instruments path/to/reference_instruments.wav
```

This will print a detailed analysis and save a visualization as `separation_analysis.png`.

## 📦 Dataset (MUSDB18)

While sample data scripts are provided, this project is designed with the [MUSDB18 dataset](https://sigsep.github.io/datasets/musdb.html) in mind for more robust training.

- Download it manually if desired.
- You will need to adapt the `converter/convert.py` script or your workflow to process the MUSDB18 structure and place the generated spectrograms in a location accessible by `train.py`.

## 📦 Dataset .h5 (MUSDB18)

Este proyecto soporta entrenamiento directamente desde archivos `.h5` con espectrogramas preprocesados (por ejemplo, MUSDB18).

- Coloca los archivos `.h5` en la carpeta `sample_data/h5/`.
- Ejemplo de ruta: `sample_data/h5/musdb18_train_spectrograms.h5`
- **No subas estos archivos al repositorio.**

Para usar el dataset `.h5` en el entrenamiento:

```python
from u_net_stft.h5_dataset import H5SpectrogramDataset
from u_net_stft.augment import spec_augment

dataset = H5SpectrogramDataset('sample_data/h5/musdb18_train_spectrograms.h5', transform=spec_augment)
```

Puedes aplicar augmentations como SpecAugment directamente sobre los espectrogramas durante el entrenamiento.

## ⏳ Status

- Core training with STFT implemented.
- STFT prediction pipeline needs implementation.
