from unittest.mock import patch

import numpy as np

from converter.convert_stems import process_file
from converter.convert_stems import stereo_to_mono, compute_normalized_stft


def test_stereo_to_mono_with_stereo_input():
    stereo_signal = np.array([[1.0, 2.0], [3.0, 4.0]])
    mono = stereo_to_mono(stereo_signal)
    expected = np.mean(stereo_signal, axis=1)
    np.testing.assert_array_almost_equal(mono, expected)


def test_stereo_to_mono_with_mono_input():
    mono_input = np.array([1.0, 2.0, 3.0])
    output = stereo_to_mono(mono_input)
    np.testing.assert_array_equal(output, mono_input)


def test_compute_normalized_stft_shape_and_range():
    sr = 22050
    t = np.linspace(0, 1, sr)
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    norm, db = compute_normalized_stft(y, sr)
    assert norm.shape == db.shape
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0


@patch("converter.convert_stems.stempeg.read_stems")
@patch("converter.convert_stems.save_waveform_image")
@patch("converter.convert_stems.save_spectrogram_image")
@patch("converter.convert_stems.np.save")
def test_process_file_runs_successfully(mock_save, mock_save_spec, mock_save_wave, mock_read_stems, tmp_path):
    fake_audio = np.random.rand(5, 44100, 2).astype(np.float32)
    sr = 44100
    mock_read_stems.return_value = (fake_audio, sr)

    test_file = tmp_path / "test.stem.mp4"
    test_file.write_text("dummy")

    process_file(test_file, tmp_path, tmp_path)

    assert mock_save.call_count == 3
    assert mock_save_spec.call_count == 3
    assert mock_save_wave.call_count == 3
