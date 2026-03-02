import os
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

# 1. SET YOUR CREDENTIALS
os.environ['KAGGLE_USERNAME'] = "smruthiravindra"
os.environ['KAGGLE_KEY'] = "7c053add8c18f856533fbdadcb8bdd1e"

def download_and_setup():
    api = KaggleApi()
    api.authenticate()
    
    # This specific dataset is ALREADY in subfolders: Filled, Default, Crossed, Invalid
    dataset_slug = "zalcode/bubble-answer-dataset"
    target_path = "dataset/train"
    
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    print(f"🚀 Downloading pre-categorized circle bubbles from {dataset_slug}...")
    
    # This command downloads AND unzips directly into the folders
    api.dataset_download_files(dataset_slug, path=target_path, unzip=True)
    
    print(f"✅ SUCCESS! Your folders are ready at: {os.path.abspath(target_path)}")
    print("📁 Folders created: Filled, Default, Crossed, Invalid")

if __name__ == "__main__":
    download_and_setup()