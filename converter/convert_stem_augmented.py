import shutil
from pathlib import Path
import random
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import stempeg
from joblib import Parallel, delayed
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from sample_downloader.download import MUSDB_SAMPLE_OUTPUT_PATH

# --- Configurable constants ---
SPECTROGRAM_OUTPUT_PATH = "{}/sample_data/spectrograms_augmented_chunks_6s"
WAVEFORM_OUTPUT_PATH = "{}/sample_data/waveforms_augmented_chunks_6s"

# Reduced parameters for memory efficiency
TARGET_SR = 22050  # Reduced sample rate (from 44100)
N_MELS = 32  # Reduced number of Mel bands (from 64)
N_FFT = 512  # Reduced FFT window size (from 1024)
HOP_LENGTH = 256  # Reduced hop length (from 512)
CHUNK_DURATION_SECONDS = 6.0  # Duration of each chunk in seconds

# Augmentation parameters
TIME_STRETCH_RATES = [0.9, 1.1]  # -10% and +10% speed
PITCH_SHIFT_STEPS = [-2, 2]  # Shift pitch up/down by 2 semitones
NOISE_LEVELS = [0.001, 0.002]  # Very small amount of noise
SEGMENT_DURATION = 5  # Duration of each segment in seconds
N_SYNTHETIC_SONGS = 2  # Number of synthetic songs to generate per training song

# --- Audio processing utilities ---
def stereo_to_mono(signal: np.ndarray) -> np.ndarray:
    """Convert stereo signal to mono by averaging channels."""
    if signal.ndim == 2 and signal.shape[1] == 2:
        return np.mean(signal, axis=1)
    return signal

def preprocess_audio(y: np.ndarray, sr: int) -> np.ndarray:
    """Preprocess audio by converting to mono and resampling."""
    # Convert to mono first
    y_mono = stereo_to_mono(y)
    # Resample to lower sample rate
    if sr != TARGET_SR:
        y_mono = librosa.resample(y=y_mono, orig_sr=sr, target_sr=TARGET_SR)
    return y_mono

def apply_time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    """Apply time stretching to audio signal."""
    return librosa.effects.time_stretch(y, rate=rate)

def apply_pitch_shift(y: np.ndarray, sr: int, steps: int) -> np.ndarray:
    """Apply pitch shifting to audio signal."""
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)

def apply_noise(y: np.ndarray, noise_level: float) -> np.ndarray:
    """Add random noise to audio signal."""
    noise = np.random.normal(0, noise_level, y.shape)
    return y + noise

def create_synthetic_song(y: np.ndarray, sr: int) -> np.ndarray:
    """Create a synthetic song by applying random augmentations to segments."""
    # Calculate number of samples per segment
    segment_length = int(SEGMENT_DURATION * sr)
    
    # Pad audio if needed to ensure it's divisible by segment length
    if len(y) % segment_length != 0:
        pad_length = segment_length - (len(y) % segment_length)
        y = np.pad(y, (0, pad_length))
    
    # Split audio into segments
    segments = np.array_split(y, len(y) // segment_length)
    augmented_segments = []
    
    for segment in segments:
        # Randomly choose an augmentation technique
        aug_choice = random.choice(['none', 'time', 'pitch', 'noise'])
        
        if aug_choice == 'none':
            augmented_segments.append(segment)
        elif aug_choice == 'time':
            rate = random.choice(TIME_STRETCH_RATES)
            try:
                stretched = apply_time_stretch(segment.copy(), rate)
                # Ensure segment length is maintained
                if len(stretched) > segment_length:
                    stretched = stretched[:segment_length]
                elif len(stretched) < segment_length:
                    stretched = np.pad(stretched, (0, segment_length - len(stretched)))
                augmented_segments.append(stretched)
            except Exception:
                augmented_segments.append(segment)  # Fallback to original if augmentation fails
        elif aug_choice == 'pitch':
            steps = random.choice(PITCH_SHIFT_STEPS)
            try:
                pitched = apply_pitch_shift(segment.copy(), sr, steps)
                augmented_segments.append(pitched)
            except Exception:
                augmented_segments.append(segment)
        else:  # noise
            level = random.choice(NOISE_LEVELS)
            try:
                noised = apply_noise(segment.copy(), level)
                augmented_segments.append(noised)
            except Exception:
                augmented_segments.append(segment)
    
    # Concatenate segments
    synthetic_song = np.concatenate(augmented_segments)
    return synthetic_song

def compute_normalized_mel(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized Mel spectrogram and dB Mel spectrogram."""
    # Add small epsilon to avoid zero values
    eps = 1e-6
    y = y + eps * np.random.randn(*y.shape)
    
    mel_spec = librosa.feature.melspectrogram(
        y=y, 
        sr=sr, 
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Safe normalization
    min_db = mel_db.min()
    max_db = mel_db.max()
    if max_db > min_db:
        mel_norm = (mel_db - min_db) / (max_db - min_db)
    else:
        # If all values are the same, return zeros
        mel_norm = np.zeros_like(mel_db)
    
    # Replace any remaining invalid values
    mel_norm = np.nan_to_num(mel_norm, nan=0.0, posinf=1.0, neginf=0.0)
    
    return mel_norm, mel_db

def compute_normalized_stft(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized STFT spectrogram and dB STFT spectrogram."""
    # Add small epsilon to avoid zero values
    eps = 1e-6
    y = y + eps * np.random.randn(*y.shape)
    
    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    magnitude, _ = librosa.magphase(stft)
    stft_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    
    # Safe normalization
    min_db = stft_db.min()
    max_db = stft_db.max()
    if max_db > min_db:
        stft_norm = (stft_db - min_db) / (max_db - min_db)
    else:
        # If all values are the same, return zeros
        stft_norm = np.zeros_like(stft_db)
    
    # Replace any remaining invalid values
    stft_norm = np.nan_to_num(stft_norm, nan=0.0, posinf=1.0, neginf=0.0)
    
    return stft_norm, stft_db

def save_spectrogram_image(spec_db: np.ndarray, save_path: Path, sr: int):
    """Plot and save spectrogram image without borders."""
    fig, ax = plt.subplots(figsize=(12, 8))
    librosa.display.specshow(spec_db, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', ax=ax)
    plt.axis('off')
    plt.margins(0)
    plt.subplots_adjust(0, 0, 1, 1)
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def save_waveform_image(signal: np.ndarray, save_path: Path, sr: int):
    """Plot and save waveform image without borders."""
    fig, ax = plt.subplots(figsize=(12, 3))
    librosa.display.waveshow(signal, sr=sr, ax=ax)
    plt.axis('off')
    plt.margins(0)
    plt.subplots_adjust(0, 0, 1, 1)
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def process_chunk(chunk: np.ndarray, sr: int, file_stem: str, chunk_idx: int, 
                 spectrogram_dir: Path, waveform_dir: Path):
    """Process a single chunk of audio data."""
    try:
        # MEL spectrograms
        mel_norm, mel_db = compute_normalized_mel(chunk, sr)
        np.save(spectrogram_dir / f"{file_stem}_chunk{chunk_idx}_mix_mel.npy", mel_norm)
        save_spectrogram_image(mel_db, spectrogram_dir / f"{file_stem}_chunk{chunk_idx}_mix_mel.png", sr)
        
        # STFT spectrograms
        stft_norm, stft_db = compute_normalized_stft(chunk, sr)
        np.save(spectrogram_dir / f"{file_stem}_chunk{chunk_idx}_mix_stft.npy", stft_norm)
        save_spectrogram_image(stft_db, spectrogram_dir / f"{file_stem}_chunk{chunk_idx}_mix_stft.png", sr)
        
        # Waveform
        save_waveform_image(chunk, waveform_dir / f"{file_stem}_chunk{chunk_idx}_mix_waveform.png", sr)
        
    except Exception as e:
        print(f"Error processing chunk {chunk_idx} of {file_stem}: {e}")

def process_file(file: Path, spectrogram_dir: Path, waveform_dir: Path):
    """Process one audio stem file: generate and save MEL, STFT, waveform outputs."""
    try:
        print(f"Processing {file.name}")
        
        # Read stems for current file
        audio, sr = stempeg.read_stems(str(file), stem_id=[0, 3, 4])
        
        # Extract and preprocess stems
        mix = preprocess_audio(audio[0], sr)
        accompaniment = preprocess_audio(audio[1], sr)
        vocals = preprocess_audio(audio[2], sr)
        
        # Update sr to new sample rate
        sr = TARGET_SR
        
        # Calculate chunk size in samples
        chunk_size = int(CHUNK_DURATION_SECONDS * sr)
        
        # Process original mix in chunks
        print(f"Processing original version of {file.name}")
        for i in range(0, len(mix), chunk_size):
            chunk = mix[i:i + chunk_size]
            
            # If the chunk is too small (last chunk), pad it
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            
            chunk_idx = i // chunk_size
            process_chunk(chunk, sr, file.stem, chunk_idx, spectrogram_dir, waveform_dir)
        
        # Create and process synthetic versions if it's a training file
        if "train" in str(file):
            print(f"Creating synthetic versions for {file.name}")
            for i in range(N_SYNTHETIC_SONGS):
                try:
                    print(f"Creating synthetic version {i+1} for {file.name}")
                    synthetic = create_synthetic_song(mix.copy(), sr)
                    
                    # Process synthetic version in chunks
                    for j in range(0, len(synthetic), chunk_size):
                        chunk = synthetic[j:j + chunk_size]
                        
                        # If the chunk is too small (last chunk), pad it
                        if len(chunk) < chunk_size:
                            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                        
                        chunk_idx = j // chunk_size
                        synthetic_name = f"{file.stem}_synthetic_{i+1}"
                        process_chunk(chunk, sr, synthetic_name, chunk_idx, spectrogram_dir, waveform_dir)
                    
                    print(f"Finished synthetic version {i+1} for {file.name}")
                except Exception as e:
                    print(f"Error creating synthetic version {i+1} for {file.name}: {e}")
        
        print(f"Finished processing {file.name}")
        
    except Exception as e:
        print(f"Error processing {file.name}: {e}")

def convert():
    """Main entry point to process all audio stems into spectrogram and waveform images."""
    project_root = Path.cwd()
    
    spectrogram_dir = Path(SPECTROGRAM_OUTPUT_PATH.format(project_root))
    waveform_dir = Path(WAVEFORM_OUTPUT_PATH.format(project_root))
    
    # Clean and recreate output folders
    for d in [spectrogram_dir, waveform_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    
    # Locate dataset
    musdb_root = Path(MUSDB_SAMPLE_OUTPUT_PATH.format(project_root))
    print(musdb_root)
    
    # Collect all files
    train_files = list((musdb_root / 'train').glob('*.stem.mp4'))
    test_files = list((musdb_root / 'test').glob('*.stem.mp4'))
    
    if not train_files and not test_files:
        print("No stem files found. Check MUSDB_SAMPLE_OUTPUT_PATH.")
        return
    
    print(f"Found {len(train_files)} training files and {len(test_files)} test files.")
    print("Starting processing with reduced parallelism...")
    
    # Process files with reduced parallelism
    n_jobs = max(1, os.cpu_count() // 2)  # Use half of available CPU cores
    
    # Process training files (with synthetic generation)
    Parallel(n_jobs=n_jobs)(
        delayed(process_file)(file, spectrogram_dir, waveform_dir)
        for file in train_files
    )
    
    # Process test files (no synthetic generation)
    Parallel(n_jobs=n_jobs)(
        delayed(process_file)(file, spectrogram_dir, waveform_dir)
        for file in test_files
    )
    
    print("All processing completed successfully.")

if __name__ == "__main__":
    convert()