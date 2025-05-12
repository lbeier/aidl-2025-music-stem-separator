import numpy as np
import pytest

from converter.convert import stereo_to_mono, compute_mel, compute_stft


@pytest.fixture
def dummy_signal():
    return np.random.randn(44100).astype(np.float32)  # 1 second of noise

def test_stereo_to_mono_average():
    stereo = np.array([[1, 2], [3, 4]], dtype=np.float32)
    expected = np.array([1.5, 3.5], dtype=np.float32)
    result = stereo_to_mono(stereo)
    assert np.allclose(result, expected), "Mono conversion failed"

def test_compute_mel_shape_and_range(dummy_signal):
    mel_norm, mel_db = compute_mel(dummy_signal, sr=44100)
    assert mel_norm.shape[0] == 128, "Expected 128 mel bands"
    assert mel_norm.min() >= 0.0 and mel_norm.max() <= 1.0, "mel_norm not in [0,1]"
    assert mel_db.min() >= -80.0 and mel_db.max() <= 0.0, "mel_db not in [-80,0]"

def test_compute_stft_shape_and_range(dummy_signal):
    stft_norm, stft_db = compute_stft(dummy_signal, sr=44100)
    assert stft_norm.shape[0] == 1025, "Expected 1025 STFT bins"
    assert stft_norm.min() >= 0.0 and stft_norm.max() <= 1.0, "stft_norm not in [0,1]"
    assert stft_db.min() >= -80.0 and stft_db.max() <= 0.0, "stft_db not in [-80,0]"
