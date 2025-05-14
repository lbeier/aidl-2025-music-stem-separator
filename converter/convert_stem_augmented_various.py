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
SPECTROGRAM_OUTPUT_PATH = "{}/sample_data/spectrograms_augmented_various"  # Different output folder
WAVEFORM_OUTPUT_PATH = "{}/sample_data/waveforms_augmented_various"  # Different output folder

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
SEGMENT_DURATION = 6  # Duration of each segment in seconds
N_SYNTHETIC_SONGS = 2  # Number of synthetic songs to generate per training song
N_SOURCE_SONGS = 3  # Number of source songs to mix for each synthetic song
CROSSFADE_DURATION = 0.1  # Crossfade duration in seconds

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

def apply_crossfade(seg1: np.ndarray, seg2: np.ndarray, fade_length: int) -> np.ndarray:
    """Apply crossfade between two audio segments."""
    if len(seg1) < fade_length or len(seg2) < fade_length:
        return np.concatenate([seg1, seg2])
    
    # Create fade curves
    fade_out = np.linspace(1.0, 0.0, fade_length)
    fade_in = np.linspace(0.0, 1.0, fade_length)
    
    # Apply crossfade
    seg1_end = seg1[:-fade_length]
    crossfade = (seg1[-fade_length:] * fade_out) + (seg2[:fade_length] * fade_in)
    seg2_start = seg2[fade_length:]
    
    return np.concatenate([seg1_end, crossfade, seg2_start])

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

def extract_segments(audio: np.ndarray, sr: int) -> list[np.ndarray]:
    """Extract fixed-length segments from audio."""
    segment_length = int(SEGMENT_DURATION * sr)
    
    # Pad audio if needed
    if len(audio) % segment_length != 0:
        pad_length = segment_length - (len(audio) % segment_length)
        audio = np.pad(audio, (0, pad_length))
    
    # Split into segments
    return np.array_split(audio, len(audio) // segment_length)

def augment_segment(segment: np.ndarray, sr: int) -> np.ndarray:
    """Apply random augmentation to a segment."""
    aug_choice = random.choice(['none', 'time', 'pitch', 'noise'])
    segment_length = len(segment)
    
    try:
        if aug_choice == 'none':
            return segment
        elif aug_choice == 'time':
            rate = random.choice(TIME_STRETCH_RATES)
            stretched = apply_time_stretch(segment, rate)
            # Maintain segment length
            if len(stretched) > segment_length:
                return stretched[:segment_length]
            else:
                return np.pad(stretched, (0, segment_length - len(stretched)))
        elif aug_choice == 'pitch':
            return apply_pitch_shift(segment, sr, random.choice(PITCH_SHIFT_STEPS))
        else:  # noise
            return apply_noise(segment, random.choice(NOISE_LEVELS))
    except Exception:
        return segment

def create_synthetic_song_from_multiple(source_songs: list[tuple[np.ndarray, int]], target_duration: int) -> np.ndarray:
    """Create a synthetic song by mixing segments from multiple songs."""
    if not source_songs:
        raise ValueError("No source songs provided")
    
    sr = source_songs[0][1]  # Sample rate from first song
    crossfade_samples = int(CROSSFADE_DURATION * sr)
    
    # Extract segments from all source songs
    all_segments = []
    for audio, _ in source_songs:
        segments = extract_segments(audio, sr)
        all_segments.extend(segments)
    
    if not all_segments:
        raise ValueError("No segments extracted from source songs")
    
    # Shuffle segments
    random.shuffle(all_segments)
    
    # Select enough segments to reach target duration
    segment_length = int(SEGMENT_DURATION * sr)
    n_segments_needed = target_duration // segment_length + 1
    selected_segments = all_segments[:n_segments_needed]
    
    # Augment selected segments
    augmented_segments = [augment_segment(seg, sr) for seg in selected_segments]
    
    # Combine segments with crossfading
    synthetic_song = augmented_segments[0]
    for i in range(1, len(augmented_segments)):
        synthetic_song = apply_crossfade(synthetic_song, augmented_segments[i], crossfade_samples)
    
    # Trim to target duration
    target_samples = target_duration * sr
    if len(synthetic_song) > target_samples:
        synthetic_song = synthetic_song[:target_samples]
    
    return synthetic_song

def compute_normalized_mel(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized Mel spectrogram and dB Mel spectrogram."""
    mel_spec = librosa.feature.melspectrogram(
        y=y, 
        sr=sr, 
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
    return mel_norm, mel_db

def compute_normalized_stft(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized STFT spectrogram and dB STFT spectrogram."""
    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    magnitude, _ = librosa.magphase(stft)
    stft_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    stft_norm = (stft_db - stft_db.min()) / (stft_db.max() - stft_db.min())
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

def process_chunk(chunk: np.ndarray, sr: int, file_stem: str, chunk_idx: int, version_name: str,
                 spectrogram_dir: Path, waveform_dir: Path):
    """Process a single chunk of audio data."""
    try:
        suffix = f"_{version_name}" if version_name != "original" else ""
        
        # MEL spectrograms
        mel_norm, mel_db = compute_normalized_mel(chunk, sr)
        np.save(spectrogram_dir / f"{file_stem}{suffix}_chunk{chunk_idx}_mix_mel.npy", mel_norm)
        save_spectrogram_image(mel_db, spectrogram_dir / f"{file_stem}{suffix}_chunk{chunk_idx}_mix_mel.png", sr)
        
        # STFT spectrograms
        stft_norm, stft_db = compute_normalized_stft(chunk, sr)
        np.save(spectrogram_dir / f"{file_stem}{suffix}_chunk{chunk_idx}_mix_stft.npy", stft_norm)
        save_spectrogram_image(stft_db, spectrogram_dir / f"{file_stem}{suffix}_chunk{chunk_idx}_mix_stft.png", sr)
        
        # Waveform
        save_waveform_image(chunk, waveform_dir / f"{file_stem}{suffix}_chunk{chunk_idx}_mix_waveform.png", sr)
        
    except Exception as e:
        print(f"Error processing chunk {chunk_idx} of {file_stem} {version_name}: {e}")

def process_audio_in_chunks(audio_data: np.ndarray, sr: int, file_stem: str, version_name: str,
                          spectrogram_dir: Path, waveform_dir: Path):
    """Process audio data in chunks."""
    # Calculate chunk size in samples
    chunk_size = int(CHUNK_DURATION_SECONDS * sr)
    
    # Process in chunks
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i + chunk_size]
        
        # If the chunk is too small (last chunk), pad it
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        
        chunk_idx = i // chunk_size
        process_chunk(chunk, sr, file_stem, chunk_idx, version_name, spectrogram_dir, waveform_dir)

def process_file(file: Path, spectrogram_dir: Path, waveform_dir: Path, all_training_files: list[Path]):
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
        
        # Process original version in chunks
        process_audio_in_chunks(mix, sr, file.stem, "original", spectrogram_dir, waveform_dir)
        
        # For training files, create synthetic versions
        if "train" in str(file):
            target_duration = len(mix) // sr  # Duration in seconds
            
            # Create synthetic versions
            for i in range(N_SYNTHETIC_SONGS):
                try:
                    # Randomly select source files for mixing
                    source_files = random.sample(all_training_files, min(N_SOURCE_SONGS, len(all_training_files)))
                    source_songs = []
                    
                    # Read stems from source files
                    for src_file in source_files:
                        src_audio, src_sr = stempeg.read_stems(str(src_file), stem_id=[0, 3, 4])
                        # Add each stem type (mix, accompaniment, vocals) as a potential source
                        source_songs.extend([
                            (preprocess_audio(src_audio[0], src_sr), TARGET_SR),
                            (preprocess_audio(src_audio[1], src_sr), TARGET_SR),
                            (preprocess_audio(src_audio[2], src_sr), TARGET_SR)
                        ])
                    
                    # Create synthetic version
                    synthetic = create_synthetic_song_from_multiple(source_songs, target_duration)
                    
                    # Process synthetic version in chunks
                    process_audio_in_chunks(synthetic, sr, file.stem, f"synthetic_mix_{i+1}", 
                                         spectrogram_dir, waveform_dir)
                    
                except Exception as e:
                    print(f"Warning: Failed to create synthetic song {i+1}: {e}")
        
        print(f"Finished {file.name}")
        
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
    
    # Process training files first (with synthetic generation)
    Parallel(n_jobs=n_jobs)(
        delayed(process_file)(file, spectrogram_dir, waveform_dir, train_files)
        for file in train_files
    )
    
    # Process test files (no synthetic generation)
    Parallel(n_jobs=n_jobs)(
        delayed(process_file)(file, spectrogram_dir, waveform_dir, train_files)
        for file in test_files
    )
    
    print("All processing completed successfully.")

if __name__ == "__main__":
    convert() 