import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import argparse


def prepare_dataset(input_dir, output_dir, sr=44100, n_mels=256, n_fft=4096, hop_length=1024):
    """
    Prepare dataset by:
    1. Converting all audio to mono
    2. Normalizing audio levels
    3. Creating high-quality spectrograms
    4. Saving in a consistent format

    Optimized for MUSDB18-HQ dataset:
    - Higher quality spectrograms (256 mel bands)
    - Larger FFT size (4096) for better frequency resolution
    - Longer hop length (1024) for better time resolution
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'spectrograms'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'waveforms'), exist_ok=True)

    # Process each song
    for song_dir in tqdm(os.listdir(input_dir)):
        if not os.path.isdir(os.path.join(input_dir, song_dir)):
            continue

        # Load stems
        mix_path = os.path.join(input_dir, song_dir, 'mixture.wav')
        vocals_path = os.path.join(input_dir, song_dir, 'vocals.wav')

        if not (os.path.exists(mix_path) and os.path.exists(vocals_path)):
            continue

        try:
            # Load and normalize audio
            mix, _ = librosa.load(mix_path, sr=sr, mono=True)
            vocals, _ = librosa.load(vocals_path, sr=sr, mono=True)

            # Normalize audio levels using peak normalization
            mix = librosa.util.normalize(mix, norm=np.inf)
            vocals = librosa.util.normalize(vocals, norm=np.inf)

            # Create high-quality spectrograms
            mix_mel = librosa.feature.melspectrogram(
                y=mix,
                sr=sr,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length,
                fmin=20,  # Lower frequency bound
                fmax=sr/2  # Upper frequency bound (Nyquist frequency)
            )
            vocals_mel = librosa.feature.melspectrogram(
                y=vocals,
                sr=sr,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length,
                fmin=20,
                fmax=sr/2
            )

            # Convert to dB scale with better reference
            mix_mel_db = librosa.power_to_db(mix_mel, ref=np.max, top_db=80)
            vocals_mel_db = librosa.power_to_db(
                vocals_mel, ref=np.max, top_db=80)

            # Save processed data
            song_name = song_dir.replace(' ', '_')
            np.save(
                os.path.join(output_dir, 'spectrograms',
                             f'{song_name}_mix.npy'),
                mix_mel_db
            )
            np.save(
                os.path.join(output_dir, 'spectrograms',
                             f'{song_name}_vocals.npy'),
                vocals_mel_db
            )
            sf.write(
                os.path.join(output_dir, 'waveforms', f'{song_name}_mix.wav'),
                mix, sr
            )
            sf.write(
                os.path.join(output_dir, 'waveforms',
                             f'{song_name}_vocals.wav'),
                vocals, sr
            )

            print(f"Processed {song_name} successfully")

        except Exception as e:
            print(f"Error processing {song_dir}: {str(e)}")
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for training")
    parser.add_argument('--input_dir', type=str, required=True,
                        help="Directory containing the raw MUSDB18-HQ dataset")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save processed data")
    parser.add_argument('--sr', type=int, default=44100,
                        help="Sample rate for audio processing (default: 44100 for HQ)")
    parser.add_argument('--n_mels', type=int, default=256,
                        help="Number of mel bands (default: 256 for better frequency resolution)")
    parser.add_argument('--n_fft', type=int, default=4096,
                        help="FFT window size (default: 4096 for better frequency resolution)")
    parser.add_argument('--hop_length', type=int, default=1024,
                        help="Number of samples between windows (default: 1024 for better time resolution)")

    args = parser.parse_args()

    print(f"Processing MUSDB18-HQ dataset from {args.input_dir}")
    print(f"Saving processed data to {args.output_dir}")
    print(f"Using parameters:")
    print(f"- Sample rate: {args.sr} Hz")
    print(f"- Mel bands: {args.n_mels}")
    print(f"- FFT size: {args.n_fft}")
    print(f"- Hop length: {args.hop_length}")

    prepare_dataset(
        args.input_dir,
        args.output_dir,
        sr=args.sr,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length
    )


if __name__ == "__main__":
    main()
