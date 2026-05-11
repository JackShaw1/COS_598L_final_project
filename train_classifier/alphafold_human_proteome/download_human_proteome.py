import tarfile
import urllib.request

"""
This script downloads the human proteome from the AlphaFold Database, which contains over 20,000 structure predictions.
This downloaded dataset is used to search for potential Imatinib binding pockets throughout the human proteome.
Eventually, we apply our trained classfier to pockets throughout this set.
Before applying our classifier, we used this dataset to simulate control binding pocket samples that were randomly selected from the surfaces of proteins.
"""

HUMAN_AF_URL = "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000005640_9606_HUMAN_v6.tar"

# Download .tar file from AlphaFold Databse
def download_file(url, output_path):
    print(f"Downloading from:\n{url}")
    urllib.request.urlretrieve(url, output_path)
    print("\nDownload complete.")
    print(f"Saved to: {output_path}")

# Extract compressed data from .tar file
def extract_tar(tar_path, extract_dir):
    print(f"Extracting {tar_path} into {extract_dir} ...")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=extract_dir)
    print("Extraction complete.")

if __name__ == "__main__":

    # Set destination path for download, and download the AlphaFold Database's human proteome containing over 20,000 structure predictions
    tar_path = "train_classifier/alphafold_human_proteome/human_proteome_AF_database.tar"
    download_file(HUMAN_AF_URL, tar_path)

    # # Extract the downloaded tar file to a directory for use with classifier
    # extract_dir = "train_classifier/alphafold_human_proteome/extracted"
    # extract_tar(tar_path, extract_dir)