# Multi-Song Mixing Data Augmentation (convert_augmented_various.py)

This script implements an advanced data augmentation technique that creates new synthetic songs by mixing segments from multiple source songs. The approach is designed to increase dataset diversity while maintaining musical coherence.

## Key Features

1. Multi-Source Mixing:
   - Takes segments from 3 different songs (configurable via N_SOURCE_SONGS)
   - Uses all stem types (mixture, accompaniment, vocals) as potential sources
   - Creates unique combinations of musical elements

2. Segmentation and Crossfading:
   - Splits songs into 3-second segments (configurable via SEGMENT_DURATION)
   - Applies smooth crossfading between segments (0.1s duration)
   - Maintains consistent audio length with the original

3. Augmentation Techniques:
   - Time stretching: ±10% speed variation
   - Pitch shifting: ±2 semitones
   - Noise addition: Very small amounts (0.001-0.002 level)
   - Random selection of augmentation per segment

4. Memory Efficiency:
   - Reduced sample rate (22050 Hz)
   - Reduced number of mel bands (32)
   - Smaller FFT window size (512)
   - Reduced hop length (256)

## Output Structure

The script generates two types of outputs in separate directories:
1. spectrograms_various/: Contains spectrograms
   - MEL spectrograms (.npy and .png)
   - STFT spectrograms (.npy and .png)
2. waveforms_various/: Contains waveform visualizations (.png)

File naming convention:
- Original: songname_original_*
- Synthetic: songname_synthetic_mix_1_*, songname_synthetic_mix_2_*

## Parameters

Key configurable parameters:
- TARGET_SR = 22050          # Sample rate
- N_MELS = 32               # Number of Mel bands
- N_FFT = 512               # FFT window size
- HOP_LENGTH = 256          # Hop length
- SEGMENT_DURATION = 3      # Seconds per segment
- N_SYNTHETIC_SONGS = 2     # Synthetic songs per original
- N_SOURCE_SONGS = 3        # Source songs to mix
- CROSSFADE_DURATION = 0.1  # Crossfade duration in seconds

## Process Flow

1. For each training song:
   a. Load the original song and its stems
   b. Randomly select 3 other source songs
   c. Extract stems from all source songs
   d. Split all stems into 3-second segments
   e. Randomly shuffle segments
   f. Apply random augmentations to selected segments
   g. Combine segments with crossfading
   h. Generate and save spectrograms and waveforms

2. For test songs:
   - Only process originals (no augmentation)
   - Maintain same output format

## Memory Optimization

- Processes files with reduced parallelism (half of CPU cores)
- Uses mono audio instead of stereo
- Applies memory-efficient preprocessing
- Saves outputs progressively to manage memory usage

## Usage

1. Ensure MUSDB dataset is available
2. Run: python convert_augmented_various.py
3. Check output in sample_data/spectrograms_various and sample_data/waveforms_various

Note: This approach creates more diverse training data by combining elements from multiple songs, which can help improve model generalization. The crossfading and careful segment handling ensure smooth transitions between different musical parts. 