import os
import sys
import shutil
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

# --- Add project root to Python path ---
# This allows importing modules from the parent directory (like constants)
project_root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root_path))

# --- Configuration ---
# Define the expected input directory structure for WAV files
# Assumes structure like: .../musdb_wav/train/SONG_TITLE/vocals.wav, etc.
MUSDB_WAV_INPUT_PATH = project_root_path / "sample_data" / "musdb_wav"

# Output paths (can be the same as the original convert script)
SPECTROGRAM_OUTPUT_PATH = project_root_path / "sample_data" / "spectrograms"
WAVEFORM_OUTPUT_PATH = project_root_path / "sample_data" / "waveforms"

# Spectrogram parameters (reuse from original script if desired)
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
CHUNK_DURATION_SECONDS = 4.0 # Duration of each chunk in seconds

# Stems to load and combine for 'instruments'
INSTRUMENT_STEMS = ['drums', 'bass', 'other']
REQUIRED_STEMS = ['mixture', 'vocals'] + INSTRUMENT_STEMS

# --- Reusable Utility Functions (Copied/adapted from convert.py) ---

def stereo_to_mono(signal: np.ndarray) -> np.ndarray:
    """Convert stereo signal (channels first or last) to mono by averaging channels."""
    if signal.ndim == 1:
        # Already mono
        return signal
    elif signal.ndim == 2:
        # Check if channels are the first dimension (e.g., shape (2, N) from librosa)
        if signal.shape[0] == 2:
            print(f"    Debug stereo_to_mono: Input shape {signal.shape}, averaging axis 0.")
            return np.mean(signal, axis=0)
        # Check if channels are the last dimension (e.g., shape (N, 2))
        elif signal.shape[-1] == 2:
            print(f"    Debug stereo_to_mono: Input shape {signal.shape}, averaging axis -1.")
            return np.mean(signal, axis=-1)
        # Check for mono signal with an extra dimension (e.g., shape (1, N) or (N, 1))
        elif signal.shape[0] == 1:
            print(f"    Debug stereo_to_mono: Input shape {signal.shape}, squeezing axis 0.")
            return signal.squeeze(0)
        elif signal.shape[-1] == 1:
            print(f"    Debug stereo_to_mono: Input shape {signal.shape}, squeezing axis -1.")
            return signal.squeeze(-1)
        else:
            # Unexpected 2D shape
            print(f"    Warning stereo_to_mono: Unexpected 2D shape {signal.shape}. Returning as is.")
            return signal
    else:
        # Handle > 2 dimensions? For now, just return
        print(f"    Warning stereo_to_mono: Input signal has unexpected ndim={signal.ndim}. Returning as is.")
        return signal

def compute_normalized_mel(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized Mel spectrogram and dB Mel spectrogram."""
    # Ensure input is mono
    y_mono = stereo_to_mono(y)
    mel_spec = librosa.feature.melspectrogram(y=y_mono, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    # Normalize dB spectrogram to [0, 1]
    min_db = mel_db.min()
    max_db = mel_db.max()
    if max_db > min_db:
        mel_norm = (mel_db - min_db) / (max_db - min_db)
    else:
        mel_norm = np.zeros_like(mel_db) # Avoid division by zero if signal is silent
    return mel_norm, mel_db


def compute_normalized_stft(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized STFT spectrogram and dB STFT spectrogram."""
    # Ensure input is mono
    y_mono = stereo_to_mono(y)
    stft = librosa.stft(y=y_mono, n_fft=N_FFT, hop_length=HOP_LENGTH)
    magnitude, _ = librosa.magphase(stft)
    stft_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    # Normalize dB spectrogram to [0, 1]
    min_db = stft_db.min()
    max_db = stft_db.max()
    if max_db > min_db:
        stft_norm = (stft_db - min_db) / (max_db - min_db)
    else:
        stft_norm = np.zeros_like(stft_db) # Avoid division by zero
    return stft_norm, stft_db

def save_spectrogram_image(spec_db: np.ndarray, save_path: Path, sr: int, is_mel: bool):
    """Plot and save spectrogram image without borders, with robust error handling."""
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8) if is_mel else (12, 6))
        
        # Handle different dimensionality - ensure we have a 2D array
        if spec_db.ndim > 2:
            # If more than 2D, take the first channel or collapse channels
            print(f"  Note: Collapsing {spec_db.ndim}D spectrogram to 2D for visualization")
            if spec_db.shape[0] <= 4:  # Likely channel-first format
                spec_display = np.mean(spec_db, axis=0)
            else:  # Try a different approach
                spec_display = spec_db.reshape(spec_db.shape[0], -1)[:, 0:1000]
        else:
            spec_display = spec_db
            
        # Simple image display - avoid librosa.display.specshow
        img = ax.imshow(spec_display, aspect='auto', origin='lower', cmap='viridis')
        
        # Remove all axes
        ax.axis('off')
        plt.margins(0)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        # Save and close
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
    except Exception as e:
        print(f"  Warning: Failed to save spectrogram image {save_path.name}: {e}")
        try:
            # Create a simple blank image as a fallback
            fig, ax = plt.subplots(figsize=(12, 8) if is_mel else (12, 6))
            ax.text(0.5, 0.5, "Spectrogram visualization failed", 
                   horizontalalignment='center', verticalalignment='center')
            ax.axis('off')
            fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
        except:
            print(f"  Error: Could not create even a placeholder image for {save_path.name}")


def save_waveform_image(signal: np.ndarray, save_path: Path, sr: int):
    """Plot and save waveform image without borders, with robust error handling."""
    try:
        # Ensure we're working with mono audio for visualization
        if signal.ndim > 1:
            if signal.shape[-1] in [1, 2, 3, 4]:  # Last dimension is channels
                signal_mono = np.mean(signal, axis=-1)
            elif signal.shape[0] in [1, 2, 3, 4]:  # First dimension is channels
                signal_mono = np.mean(signal, axis=0)
            else:
                # Unknown format, try to flatten or use first channel
                signal_mono = signal.flatten() if signal.size < 5000000 else signal.reshape(-1)[0:5000000]
        else:
            signal_mono = signal
            
        # Create a simple time axis based on the length and sample rate
        # This avoids librosa.display.waveshow entirely
        samples = len(signal_mono)
        
        # Downsample very long signals for visualization (max ~10k points)
        max_points = 10000
        if samples > max_points:
            # Calculate downsample factor and apply
            downsample = samples // max_points
            signal_mono = signal_mono[::downsample]
            samples = len(signal_mono)
            # Adjust time axis to reflect original duration
            time = np.linspace(0, len(signal_mono) * downsample / sr, samples)
        else:
            time = np.linspace(0, samples / sr, samples)
        
        # Use basic matplotlib plotting instead of librosa.display.waveshow
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(time, signal_mono, color='blue', linewidth=0.5)
        
        # Adjust plot appearance
        ax.set_xlim(0, time[-1])
        y_max = np.max(np.abs(signal_mono)) * 1.1  # Add 10% margin
        ax.set_ylim(-y_max, y_max)
        
        # Turn off all axes elements
        ax.axis('off')
        plt.margins(0)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        # Save the image and close
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
    except Exception as e:
        print(f"  Warning: Failed to save waveform image {save_path.name}: {e}")
        # Create an empty placeholder image to avoid missing files
        try:
            # Create a simple blank image as a fallback
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.text(0.5, 0.5, "Waveform visualization failed", 
                   horizontalalignment='center', verticalalignment='center')
            ax.axis('off')
            fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
        except:
            print(f"  Error: Could not create even a placeholder image for {save_path.name}")

# --- Per-track directory processor ---
def process_track_directory(track_dir: Path, spectrogram_dir: Path, waveform_dir: Path):
    """
    Processes a directory containing individual WAV stems for a single track.
    Loads stems, combines instruments, calculates spectrograms, and saves outputs.
    """
    track_name = track_dir.name
    print(f"Processing track: {track_name}")

    stems = {}
    target_sr = None

    # Load required stems
    try:
        for stem_name in REQUIRED_STEMS:
            wav_path = track_dir / f"{stem_name}.wav"
            if not wav_path.exists():
                print(f"  Warning: Missing stem '{stem_name}.wav' for track {track_name}. Skipping track.")
                return # Skip this track if essential stems are missing

            y, sr = librosa.load(wav_path, sr=None, mono=False) # Load as potentially stereo, sr=None to preserve original SR

            if target_sr is None:
                target_sr = sr
            elif sr != target_sr:
                print(f"  Warning: Sample rate mismatch in {track_name} ({stem_name}.wav has {sr} Hz, expected {target_sr} Hz). Skipping track.")
                return # Skip if sample rates don't match

            stems[stem_name] = y

        # <<<--- ADD DEBUGGING HERE --- >>>
        # --- Debug: Check loaded audio duration ---
        mixture_duration_samples = stems['mixture'].shape[0]
        mixture_duration_seconds = librosa.get_duration(y=stems['mixture'], sr=target_sr)
        print(f"  Debug: Loaded 'mixture.wav' - Duration: {mixture_duration_seconds:.2f}s, Samples: {mixture_duration_samples}, Shape: {stems['mixture'].shape}")
        # --- End Debug ---

        # Combine instruments
        # Ensure all instrument stems are the same length (pad if necessary)
        max_len = max(stems[name].shape[0] for name in INSTRUMENT_STEMS)
        instruments_signal = np.zeros((max_len, stems[INSTRUMENT_STEMS[0]].shape[1] if stems[INSTRUMENT_STEMS[0]].ndim > 1 else 1 ), dtype=np.float32)

        for name in INSTRUMENT_STEMS:
            signal = stems[name]
            if signal.shape[0] < max_len:
                 # Pad if shorter (axis 0 is time)
                 pad_width = ((0, max_len - signal.shape[0]), (0, 0)) if signal.ndim > 1 else ((0, max_len - signal.shape[0]))
                 signal = np.pad(signal, pad_width, mode='constant')
            instruments_signal += signal

        # Get mix and vocals (ensure they are also padded/truncated to max_len if needed)
        mix_signal = stems['mixture']
        vocals_signal = stems['vocals']
        if mix_signal.shape[0] != max_len:
            pad_width = ((0, max_len - mix_signal.shape[0]), (0, 0)) if mix_signal.ndim > 1 else ((0, max_len - mix_signal.shape[0]))
            mix_signal = np.pad(mix_signal, pad_width, mode='constant') if mix_signal.shape[0] < max_len else mix_signal[:max_len]
        if vocals_signal.shape[0] != max_len:
             pad_width = ((0, max_len - vocals_signal.shape[0]), (0, 0)) if vocals_signal.ndim > 1 else ((0, max_len - vocals_signal.shape[0]))
             vocals_signal = np.pad(vocals_signal, pad_width, mode='constant') if vocals_signal.shape[0] < max_len else vocals_signal[:max_len]

        # <<< --- Force MONO Conversion Explicitly BEFORE Spectrogram Calculation --- >>>
        mix_signal_mono = stereo_to_mono(mix_signal)
        vocals_signal_mono = stereo_to_mono(vocals_signal)
        instruments_signal_mono = stereo_to_mono(instruments_signal)
        # --- End Force Mono ---

        # --- Generate Spectrograms (using MONO versions explicitly) ---
        sr = target_sr # Use the confirmed sample rate

        # --- Calculate full spectrograms first ---
        # Pass the explicitly mono signals
        # MEL
        mix_mel_norm_full, mix_mel_db_full = compute_normalized_mel(mix_signal_mono, sr)
        vocals_mel_norm_full, vocals_mel_db_full = compute_normalized_mel(vocals_signal_mono, sr)
        instruments_mel_norm_full, instruments_mel_db_full = compute_normalized_mel(instruments_signal_mono, sr)

        # STFT
        mix_stft_norm_full, mix_stft_db_full = compute_normalized_stft(mix_signal_mono, sr)
        vocals_stft_norm_full, vocals_stft_db_full = compute_normalized_stft(vocals_signal_mono, sr)
        instruments_stft_norm_full, instruments_stft_db_full = compute_normalized_stft(instruments_signal_mono, sr)

        # --- Add Sanity Check: Ensure Spectrograms are 2D ---
        if mix_mel_norm_full.ndim != 2 or vocals_mel_norm_full.ndim != 2 or instruments_mel_norm_full.ndim != 2 or \
           mix_stft_norm_full.ndim != 2 or vocals_stft_norm_full.ndim != 2 or instruments_stft_norm_full.ndim != 2:
            print(f"  ################# ERROR ##################")
            print(f"  Spectrogram calculation did not result in 2D arrays for track {track_name}.")
            print(f"  Shapes - MEL (Mix/Voc/Inst): {mix_mel_norm_full.shape}, {vocals_mel_norm_full.shape}, {instruments_mel_norm_full.shape}")
            print(f"  Shapes - STFT(Mix/Voc/Inst): {mix_stft_norm_full.shape}, {vocals_stft_norm_full.shape}, {instruments_stft_norm_full.shape}")
            print(f"  Skipping chunking and saving for this track.")
            print(f"  ##########################################")
            return # Stop processing this track
        # --- End Sanity Check ---

        # --- Chunking Logic (Simplified assuming 2D input) ---
        # Spectrograms are now guaranteed to be 2D (freq, time)
        time_axis = 1
        frames_dim = mix_mel_norm_full.shape[time_axis]

        print(f"  Full spectrogram shape: {mix_mel_norm_full.shape}") # Should show (freq, time)

        chunk_length_frames = librosa.time_to_frames(CHUNK_DURATION_SECONDS, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT)
        num_chunks = frames_dim // chunk_length_frames

        print(f"  Track length: {frames_dim} frames. Chunk length: {chunk_length_frames} frames. Num chunks: {num_chunks}")

        if num_chunks == 0:
             print(f"  Warning: Track {track_name} is shorter than chunk duration ({CHUNK_DURATION_SECONDS}s). Skipping.")
             return

        # --- Save Chunked Outputs ---
        for i in range(num_chunks):
            start_frame = i * chunk_length_frames
            end_frame = start_frame + chunk_length_frames

            # Slicing is now always for 2D (freq, time)
            mix_mel_chunk = mix_mel_norm_full[:, start_frame:end_frame]
            vocals_mel_chunk = vocals_mel_norm_full[:, start_frame:end_frame]
            instruments_mel_chunk = instruments_mel_norm_full[:, start_frame:end_frame]

            mix_stft_chunk = mix_stft_norm_full[:, start_frame:end_frame]
            vocals_stft_chunk = vocals_stft_norm_full[:, start_frame:end_frame]
            instruments_stft_chunk = instruments_stft_norm_full[:, start_frame:end_frame]

            # --- Save NPY Chunks ---
            chunk_suffix = f"chunk_{i:04d}" # Pad chunk index e.g., chunk_0000

            # Ensure saved chunks are indeed 2D
            assert mix_mel_chunk.ndim == 2, f"Mix Mel chunk {i} is not 2D!"
            # Add more asserts if needed

            np.save(spectrogram_dir / f"{track_name}_mix_mel_{chunk_suffix}.npy", mix_mel_chunk)
            np.save(spectrogram_dir / f"{track_name}_vocals_mel_{chunk_suffix}.npy", vocals_mel_chunk)
            np.save(spectrogram_dir / f"{track_name}_instruments_mel_{chunk_suffix}.npy", instruments_mel_chunk)

            np.save(spectrogram_dir / f"{track_name}_mix_stft_{chunk_suffix}.npy", mix_stft_chunk)
            np.save(spectrogram_dir / f"{track_name}_vocals_stft_{chunk_suffix}.npy", vocals_stft_chunk)
            np.save(spectrogram_dir / f"{track_name}_instruments_stft_{chunk_suffix}.npy", instruments_stft_chunk)

        # <<<--- ADD VALIDATION HERE --- >>>
        # --- Validation: Compare total chunk duration to loaded duration ---
        total_chunk_frames = num_chunks * chunk_length_frames
        # Calculate duration from frames using librosa's utility function
        total_chunk_duration_seconds = librosa.frames_to_time(total_chunk_frames, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT)
        # Alternatively, calculate directly from samples if preferred:
        # total_chunk_duration_seconds_alt = total_chunk_frames * HOP_LENGTH / sr

        print(f"  Validation: Total duration covered by chunks: {total_chunk_duration_seconds:.2f}s.")
        duration_discrepancy_seconds = abs(mixture_duration_seconds - total_chunk_duration_seconds)

        # Allow for a small discrepancy (e.g., less than the duration of one chunk)
        if duration_discrepancy_seconds > CHUNK_DURATION_SECONDS * 1.1: # Allow 10% extra margin
             print(f"  ############# WARNING #############")
             print(f"  Significant duration discrepancy ({duration_discrepancy_seconds:.2f}s) between:")
             print(f"     - Loaded audio duration: {mixture_duration_seconds:.2f}s")
             print(f"     - Total chunk duration: {total_chunk_duration_seconds:.2f}s")
             print(f"  This might indicate an issue with loading or chunking for track {track_name}.")
             print(f"  ###################################")
        # --- End Validation ---

        # --- Optional: Keep saving full PNGs for overall visualization, or remove this section ---
        # --- It's usually better to visualize full tracks, not chunks ---

        # PNG (Full Spectrogram Images - dB Scale) - Keep or remove as desired
        try:
            save_spectrogram_image(mix_mel_db_full, spectrogram_dir / f"{track_name}_mix_mel_FULL.png", sr, is_mel=True)
            save_spectrogram_image(vocals_mel_db_full, spectrogram_dir / f"{track_name}_vocals_mel_FULL.png", sr, is_mel=True)
            save_spectrogram_image(instruments_mel_db_full, spectrogram_dir / f"{track_name}_instruments_mel_FULL.png", sr, is_mel=True)

            save_spectrogram_image(mix_stft_db_full, spectrogram_dir / f"{track_name}_mix_stft_FULL.png", sr, is_mel=False)
            save_spectrogram_image(vocals_stft_db_full, spectrogram_dir / f"{track_name}_vocals_stft_FULL.png", sr, is_mel=False)
            save_spectrogram_image(instruments_stft_db_full, spectrogram_dir / f"{track_name}_instruments_stft_FULL.png", sr, is_mel=False)
        except Exception as e:
            print(f"  Warning: FULL Spectrogram visualization failed for {track_name}: {e}")

        # PNG (Full Waveform Images - Mono) - Keep or remove as desired
        try:
            save_waveform_image(mix_signal, waveform_dir / f"{track_name}_mix_waveform_FULL.png", sr)
            save_waveform_image(vocals_signal, waveform_dir / f"{track_name}_vocals_waveform_FULL.png", sr)
            save_waveform_image(instruments_signal, waveform_dir / f"{track_name}_instruments_waveform_FULL.png", sr)
        except Exception as e:
            print(f"  Warning: FULL Waveform visualization failed for {track_name}: {e}")
            # Continue processing - waveforms aren't critical for spectrogram training

        print(f"  Finished {track_name} ({num_chunks} chunks saved)")

    except Exception as e:
        print(f"  Error processing track {track_name}: {e}")


# --- Main Processing Function ---
def convert_wav_main():
    """
    Main entry point to find track directories with WAV stems and process them
    in parallel into spectrogram and waveform outputs.
    """
    print("Starting WAV stem conversion process...")
    print(f"Expecting WAV stems in subdirectories under: {MUSDB_WAV_INPUT_PATH}")
    print(f"Spectrogram outputs (.npy, .png) will be saved to: {SPECTROGRAM_OUTPUT_PATH}")
    print(f"Waveform images (.png) will be saved to: {WAVEFORM_OUTPUT_PATH}")

    # Ensure output directories exist and are empty
    for d in [SPECTROGRAM_OUTPUT_PATH, WAVEFORM_OUTPUT_PATH]:
        if d.exists():
            print(f"Cleaning existing output directory: {d}")
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # Find all track directories in train and test sets
    track_dirs = []
    if not MUSDB_WAV_INPUT_PATH.exists():
         print(f"Error: Input WAV directory not found: {MUSDB_WAV_INPUT_PATH}")
         return

    for split in ['train', 'test']:
        split_dir = MUSDB_WAV_INPUT_PATH / split
        if split_dir.exists():
            for item in split_dir.iterdir():
                if item.is_dir(): # Check if it's a directory
                    track_dirs.append(item)
        else:
            print(f"Warning: {split} directory not found in {MUSDB_WAV_INPUT_PATH}")

    if not track_dirs:
        print("No track directories found. Please check the structure of MUSDB_WAV_INPUT_PATH.")
        print(f"Expected structure: {MUSDB_WAV_INPUT_PATH}/train/TRACK_NAME/vocals.wav etc.")
        return

    print(f"Found {len(track_dirs)} track directories. Starting parallel processing...")

    # Parallel execution across all CPU cores (-1 uses all available)
    # Consider setting n_jobs to a specific number (e.g., os.cpu_count() - 2) if needed
    Parallel(n_jobs=-1)(
        delayed(process_track_directory)(track_dir, SPECTROGRAM_OUTPUT_PATH, WAVEFORM_OUTPUT_PATH)
        for track_dir in track_dirs
    )

    print("\nAll processing completed.")


if __name__ == "__main__":
    convert_wav_main()
