import os
import subprocess
import requests
import zipfile
import gzip
import shutil

DATASETS = {

    "lecturebank":
    "https://github.com/Yale-LILY/LectureBank/archive/refs/heads/master.zip",

    "docbank":
    "https://github.com/doc-analysis/DocBank/archive/refs/heads/master.zip",

    "grotoap2":
    "https://github.com/zejiangh/GROTOAP2/archive/refs/heads/master.zip",

    "s2orc":
    "https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/s2orc/s2orc.jsonl.gz"
}

RAW_DIR = "datasets_raw"
PROCESSED_DIR = "datasets_processed"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


# -----------------------------------------
# Download file
# -----------------------------------------

def download_file(url, filename):

    print("Downloading:", url)

    r = requests.get(url, stream=True)

    with open(filename, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    print("Saved:", filename)


# -----------------------------------------
# Extract ZIP
# -----------------------------------------

def extract_zip(path):

    print("Extracting ZIP:", path)

    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(RAW_DIR)

    print("Extraction complete")


# -----------------------------------------
# Extract GZ
# -----------------------------------------

def extract_gz(path):

    print("Extracting GZ:", path)

    output = path.replace(".gz", "")

    with gzip.open(path, 'rb') as f_in:
        with open(output, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    print("Extraction complete")


# -----------------------------------------
# Download datasets
# -----------------------------------------

def download_datasets():

    for name, url in DATASETS.items():

        if url.endswith(".zip"):
            filename = os.path.join(RAW_DIR, name + ".zip")

        elif url.endswith(".gz"):
            filename = os.path.join(RAW_DIR, name + ".gz")

        else:
            filename = os.path.join(RAW_DIR, name)

        download_file(url, filename)

        if filename.endswith(".zip"):
            extract_zip(filename)

        elif filename.endswith(".gz"):
            extract_gz(filename)


# -----------------------------------------
# Run conversion scripts
# -----------------------------------------

def run_conversions():

    scripts = [
        "convert_lecturebank.py",
        "convert_docbank.py",
        "convert_grotoap2.py",
        "convert_s2orc.py"
    ]

    for script in scripts:
        print("Running:", script)
        subprocess.run(["python", script])


# -----------------------------------------
# Main
# -----------------------------------------

if __name__ == "__main__":

    print("Step 1: Download datasets")
    download_datasets()

    print("Step 2: Convert datasets")
    run_conversions()

    print("All datasets prepared successfully")