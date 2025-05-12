# Single-Song Data Augmentation (convert_augmented.py)

This script implements a data augmentation technique that creates new synthetic songs by modifying segments of individual songs while preserving their musical structure and coherence.

## Key Features

1. Single-Source Processing:
   - Works with one song at a time
   - Preserves the original musical structure
   - Maintains temporal relationships between segments
   - Processes only needed stems (mixture, accompaniment, vocals)

2. Segmentation Approach:
   - Splits songs into 3-second segments (configurable via SEGMENT_DURATION)
   - Processes each segment independently
   - Maintains original segment order
   - Preserves musical flow and progression

3. Augmentation Techniques:
   - Time stretching: ±10% speed variation
   - Pitch shifting: ±2 semitones
   - Noise addition: Very small amounts (0.001-0.002 level)
   - Random selection of augmentation per segment
   - Original segments are preserved when augmentation fails

4. Memory Efficiency:
   - Reduced sample rate (22050 Hz)
   - Reduced number of mel bands (32)
   - Smaller FFT window size (512)
   - Reduced hop length (256)

## Output Structure

The script generates two types of outputs in separate directories:
1. spectrograms/: Contains spectrograms
   - MEL spectrograms (.npy and .png)
   - STFT spectrograms (.npy and .png)
2. waveforms/: Contains waveform visualizations (.png)

File naming convention:
- Original: songname_original_*
- Synthetic: songname_synthetic_1_*, songname_synthetic_2_*

## Parameters

Key configurable parameters:
- TARGET_SR = 22050          # Sample rate
- N_MELS = 32               # Number of Mel bands
- N_FFT = 512               # FFT window size
- HOP_LENGTH = 256          # Hop length
- SEGMENT_DURATION = 3      # Seconds per segment
- N_SYNTHETIC_SONGS = 2     # Synthetic songs per original
- TIME_STRETCH_RATES = [0.9, 1.1]  # Speed variation
- PITCH_SHIFT_STEPS = [-2, 2]      # Pitch variation
- NOISE_LEVELS = [0.001, 0.002]    # Noise variation

## Process Flow

1. For each training song:
   a. Load the original song and its stems
   b. Convert to mono and resample
   c. Split into fixed-length segments
   d. For each synthetic version:
      - Process each segment independently
      - Randomly choose augmentation type
      - Apply selected augmentation
      - Concatenate processed segments
   e. Generate and save spectrograms and waveforms

2. For test songs:
   - Only process originals (no augmentation)
   - Maintain same output format

## Memory Optimization

- Processes files with reduced parallelism (half of CPU cores)
- Uses mono audio instead of stereo
- Processes one segment at a time
- Saves outputs progressively
- Handles memory-efficient preprocessing

## Advantages of Single-Song Approach

1. Musical Coherence:
   - Maintains original song structure
   - Preserves musical transitions
   - Keeps temporal relationships intact

2. Controlled Variation:
   - Predictable modifications
   - Consistent musical style
   - Maintains original song length

3. Training Benefits:
   - Creates realistic variations
   - Helps model learn invariance to specific transformations
   - Maintains semantic meaning of the original

## Usage

1. Ensure MUSDB dataset is available
2. Run: python convert_augmented.py
3. Check output in sample_data/spectrograms and sample_data/waveforms

Note: This approach is particularly useful for teaching the model robustness to specific audio transformations while maintaining the original musical structure. It's ideal for learning style-preserving separations. 