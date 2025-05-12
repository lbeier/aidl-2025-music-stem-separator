import os
import stempeg
import soundfile as sf
from pathlib import Path


def extract_stems(input_file: str, output_dir: str):
    """Extract stems from a .stem.mp4 file and save them as WAV files."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read stems
    stems, sr = stempeg.read_stems(input_file)

    # Extract and save each stem
    # stems[0] = mix
    # stems[1] = drums
    # stems[2] = bass
    # stems[3] = other
    # stems[4] = vocals

    # Save mix
    sf.write(os.path.join(output_dir, 'mix.wav'), stems[0], sr)

    # Save vocals
    sf.write(os.path.join(output_dir, 'vocals.wav'), stems[4], sr)

    # Combine and save instruments (drums + bass + other)
    instruments = stems[1] + stems[2] + stems[3]
    sf.write(os.path.join(output_dir, 'instruments.wav'), instruments, sr)

    print(f"Stems extracted and saved to {output_dir}")


if __name__ == "__main__":
    input_file = "sample_data/musdb/test/Tom McKenzie - Directions.stem.mp4"
    output_dir = "sample_data/musdb/test/stems"
    extract_stems(input_file, output_dir)
