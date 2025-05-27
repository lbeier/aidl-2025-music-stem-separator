import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal
import mir_eval.separation as mir_sep
import librosa.display
import argparse


def load_audio(filepath, sr=44100):
    """Load audio file and ensure correct sample rate."""
    try:
        y, sr_loaded = librosa.load(filepath, sr=sr, mono=True)
        return y
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def compute_spectrogram(audio, sr=44100, n_fft=2048, hop_length=512):
    """Compute magnitude spectrogram."""
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    return np.abs(D)


def compute_mel_spectrogram(audio, sr=44100, n_mels=128, n_fft=2048, hop_length=512):
    """Compute mel spectrogram."""
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    return librosa.power_to_db(mel_spec, ref=np.max)


def analyze_audio_separation(mix_path, vocals_path, instruments_path, reference_vocals_path=None, reference_instruments_path=None, sr=44100):
    """Comprehensive analysis of audio separation results."""
    print("\n=== Audio Separation Analysis ===\n")

    # Load audio files
    print("Loading audio files...")
    mix = load_audio(mix_path, sr=sr)
    vocals = load_audio(vocals_path, sr=sr)
    instruments = load_audio(instruments_path, sr=sr)

    # Load reference files if provided
    ref_vocals = load_audio(reference_vocals_path,
                            sr=sr) if reference_vocals_path else None
    ref_instruments = load_audio(
        reference_instruments_path, sr=sr) if reference_instruments_path else None

    # Find minimum length for all loaded signals
    signals = [x for x in [mix, vocals, instruments,
                           ref_vocals, ref_instruments] if x is not None]
    min_len = min(len(x) for x in signals)
    if min_len == 0:
        print("Error: One or more audio files are empty.")
        return
    # Trim all signals to min_len
    if mix is not None:
        mix = mix[:min_len]
    if vocals is not None:
        vocals = vocals[:min_len]
    if instruments is not None:
        instruments = instruments[:min_len]
    if ref_vocals is not None:
        ref_vocals = ref_vocals[:min_len]
    if ref_instruments is not None:
        ref_instruments = ref_instruments[:min_len]

    if mix is None or vocals is None or instruments is None:
        print("Error: Failed to load one or more audio files.")
        return

    # Basic audio statistics
    print("\n1. Basic Audio Statistics:")
    print(f"Mix duration: {len(mix)/sr:.2f} seconds")
    print(f"Mix RMS: {np.sqrt(np.mean(mix**2)):.4f}")
    print(f"Vocals RMS: {np.sqrt(np.mean(vocals**2)):.4f}")
    print(f"Instruments RMS: {np.sqrt(np.mean(instruments**2)):.4f}")

    # Spectral analysis
    print("\n2. Spectral Analysis:")
    mix_spec = compute_spectrogram(mix, sr=sr)
    vocals_spec = compute_spectrogram(vocals, sr=sr)
    instruments_spec = compute_spectrogram(instruments, sr=sr)

    # Frequency distribution
    mix_freq = np.mean(mix_spec, axis=1)
    vocals_freq = np.mean(vocals_spec, axis=1)
    instruments_freq = np.mean(instruments_spec, axis=1)

    freq_res = sr / 2048
    print(f"Mix frequency range: {np.argmax(mix_freq)*freq_res:.1f} Hz")
    print(f"Vocals frequency range: {np.argmax(vocals_freq)*freq_res:.1f} Hz")
    print(
        f"Instruments frequency range: {np.argmax(instruments_freq)*freq_res:.1f} Hz")

    # Phase correlation
    print("\n3. Phase Correlation Analysis:")
    mix_phase = np.angle(librosa.stft(mix))
    vocals_phase = np.angle(librosa.stft(vocals))
    instruments_phase = np.angle(librosa.stft(instruments))

    phase_corr = np.corrcoef(mix_phase.flatten(), vocals_phase.flatten())[0, 1]
    print(f"Phase correlation between mix and vocals: {phase_corr:.4f}")

    # Spectral leakage
    print("\n4. Spectral Leakage Analysis:")
    leakage = np.mean(np.abs(vocals_spec * instruments_spec))
    print(f"Spectral leakage between vocals and instruments: {leakage:.4f}")

    # SDR calculation if reference files are provided
    sdr_vocals = sdr_instruments = None
    if ref_vocals is not None and ref_instruments is not None:
        print("\n5. Signal-to-Distortion Ratio (SDR):")
        try:
            sdr_vocals, _, _, _ = mir_sep.bss_eval_sources(ref_vocals, vocals)
            sdr_instruments, _, _, _ = mir_sep.bss_eval_sources(
                ref_instruments, instruments)
            print(f"SDR for vocals: {sdr_vocals[0]:.2f} dB")
            print(f"SDR for instruments: {sdr_instruments[0]:.2f} dB")
        except Exception as e:
            print(f"Error calculating SDR: {e}")

    # Generate visualization
    print("\nGenerating visualizations...")
    plt.figure(figsize=(15, 10))

    # Waveform plot
    plt.subplot(3, 1, 1)
    plt.plot(mix[:sr])  # First second
    plt.title('Mix Waveform (First Second)')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    # Spectrogram plot
    plt.subplot(3, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(mix_spec, ref=np.max),
                             sr=sr, hop_length=512, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mix Spectrogram')

    # Mel spectrogram plot
    plt.subplot(3, 1, 3)
    mel_spec = compute_mel_spectrogram(mix, sr=sr)
    librosa.display.specshow(mel_spec, sr=sr, hop_length=512,
                             x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mix Mel Spectrogram')

    plt.tight_layout()
    plt.savefig('separation_analysis.png')
    print("Analysis visualization saved as 'separation_analysis.png'")

    # Generate summary report
    print("\n=== Separation Quality Summary ===")
    print("1. Signal Quality:")
    print(
        f"   - Vocals to instruments ratio: {np.sqrt(np.mean(vocals**2))/np.sqrt(np.mean(instruments**2)):.2f}")
    print(f"   - Spectral leakage: {leakage:.4f}")
    print(f"   - Phase correlation: {phase_corr:.4f}")

    if sdr_vocals is not None and sdr_instruments is not None:
        print("\n2. Separation Accuracy:")
        print(f"   - Vocals SDR: {sdr_vocals[0]:.2f} dB")
        print(f"   - Instruments SDR: {sdr_instruments[0]:.2f} dB")

    print("\n3. Recommendations:")
    if leakage > 0.1:
        print("   - High spectral leakage detected. Consider adjusting the model's frequency resolution.")
    if phase_corr < 0.5:
        print("   - Low phase correlation. The model might need better phase preservation.")
    if np.sqrt(np.mean(vocals**2))/np.sqrt(np.mean(instruments**2)) < 0.5:
        print("   - Vocals might be too quiet compared to instruments. Consider adjusting the balance.")

    # Additional recommendations based on SDR
    if sdr_vocals is not None and sdr_instruments is not None:
        if sdr_vocals[0] < 5.0:
            print(
                "   - Low SDR for vocals. Consider improving the model's ability to separate vocals.")
        if sdr_instruments[0] < 5.0:
            print(
                "   - Low SDR for instruments. Consider improving the model's ability to separate instruments.")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze audio separation results')
    parser.add_argument('--mix', type=str, required=True,
                        help='Path to mix audio file')
    parser.add_argument('--vocals', type=str, required=True,
                        help='Path to separated vocals file')
    parser.add_argument('--instruments', type=str, required=True,
                        help='Path to separated instruments file')
    parser.add_argument('--ref_vocals', type=str,
                        help='Path to reference vocals file (optional)')
    parser.add_argument('--ref_instruments', type=str,
                        help='Path to reference instruments file (optional)')
    parser.add_argument('--sr', type=int, default=44100,
                        help='Sample rate for analysis (default: 44100)')

    args = parser.parse_args()

    analyze_audio_separation(
        args.mix,
        args.vocals,
        args.instruments,
        args.ref_vocals,
        args.ref_instruments,
        sr=args.sr
    )


if __name__ == '__main__':
    main()
