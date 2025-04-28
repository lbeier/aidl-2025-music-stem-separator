from pathlib import Path

import musdb

PROJECT_ROOT = Path.cwd()
MUSDB_SAMPLE_OUTPUT_PATH="{}/sample_data/musdb"

def download():
    print("Downloading 7 seconds samples from MUSDB18...")

    project_root = Path.cwd()
    output_dir = Path(MUSDB_SAMPLE_OUTPUT_PATH.format(project_root))
    output_dir.mkdir(parents=True, exist_ok=True)

    musdb.DB(root=output_dir, download=True)

if __name__ == "__main__":
    download()
