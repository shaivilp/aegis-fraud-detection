"""Download the two labeled Kaggle files into ieee_data/. The data is never committed.

    pip install kaggle
    python download_data.py

You need a Kaggle API token in ~/.kaggle/kaggle.json, and you must have accepted the
competition rules at https://www.kaggle.com/competitions/ieee-fraud-detection/rules
"""
import subprocess
import zipfile

from fraud import DATA_DIR

COMPETITION = "ieee-fraud-detection"
FILES = ["train_transaction.csv", "train_identity.csv"]

DATA_DIR.mkdir(exist_ok=True)
for name in FILES:
    if (DATA_DIR / name).exists():
        print(f"{name}: already there")
        continue
    subprocess.run(["kaggle", "competitions", "download", "-c", COMPETITION, "-f", name, "-p", str(DATA_DIR)],
                   check=True)
    archive = DATA_DIR / f"{name}.zip"  # Kaggle sends large files zipped
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            z.extract(name, DATA_DIR)
        archive.unlink()
    print(f"{name}: downloaded")
