from pathlib import Path

import musdb

PROJECT_ROOT = Path.cwd()
MUSDB_SAMPLE_OUTPUT_PATH="{}/sample_data/musdb"

def download():
    print("Downloading full MUSDB18-HQ dataset (approximately 85GB)...")
    print("This could take a significant amount of time depending on your connection speed...")

    project_root = Path.cwd()
    output_dir = Path(MUSDB_SAMPLE_OUTPUT_PATH.format(project_root))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setting is_wav=True to get full-quality audio files
    # Setting subset=None ensures both 'train' and 'test' sets are downloaded
    musdb.DB(root=output_dir, download=True, is_wav=True, subsets=None)
    
    print("Download complete. Full dataset available at:", output_dir)

if __name__ == "__main__":
    download()
