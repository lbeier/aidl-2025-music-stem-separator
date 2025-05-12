import librosa
import numpy as np
import soundfile as sf
from pathlib import Path


def analyze_audio(file_path):
    """Analyze an audio file and return key metrics."""
    # Load audio
    y, sr = librosa.load(file_path, sr=None)

    # Calculate metrics
    duration = librosa.get_duration(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)[0].mean()
    zero_crossings = librosa.feature.zero_crossing_rate(y=y)[0].mean()
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()

    return {
        'duration': duration,
        'rms': rms,
        'zero_crossings': zero_crossings,
        'spectral_centroid': spectral_centroid,
        'max_amplitude': np.max(np.abs(y)),
        'mean_amplitude': np.mean(np.abs(y))
    }


def main():
    files = [
        'output/vocals_mel.wav',
        'output/instruments_mel.wav',
        'output/vocals_stft.wav',
        'output/instruments_stft.wav'
    ]

    print("Analyzing output files...\n")

    for file_path in files:
        print(f"\nAnalyzing {file_path}:")
        metrics = analyze_audio(file_path)
        print(f"Duration: {metrics['duration']:.2f} seconds")
        print(f"RMS Energy: {metrics['rms']:.6f}")
        print(f"Zero Crossing Rate: {metrics['zero_crossings']:.6f}")
        print(f"Spectral Centroid: {metrics['spectral_centroid']:.2f} Hz")
        print(f"Max Amplitude: {metrics['max_amplitude']:.6f}")
        print(f"Mean Amplitude: {metrics['mean_amplitude']:.6f}")


if __name__ == "__main__":
    main()
