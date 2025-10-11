#!/usr/bin/env python3
"""
Update the README.md on HuggingFace dataset repository
"""

from huggingface_hub import HfApi, upload_file
import os

# Configuration
HF_TOKEN = os.environ.get("HF_TOKEN")  # Set this environment variable with your HuggingFace token
DATASET_NAME = "Belief-and-Biases-Bench"
README_PATH = "hf_dataset/README.md"

if not HF_TOKEN:
    print("Error: Please set the HF_TOKEN environment variable with your HuggingFace token")
    print("Example: export HF_TOKEN='your_token_here'")
    exit(1)

def main():
    # Initialize HuggingFace API
    api = HfApi(token=HF_TOKEN)
    
    # Get the username
    whoami = api.whoami()
    username = whoami['name']
    
    repo_id = f"{username}/{DATASET_NAME}"
    
    print(f"Updating README.md for {repo_id}...")
    
    # Upload the README file
    try:
        upload_file(
            path_or_fileobj=README_PATH,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=HF_TOKEN,
            commit_message="Update dataset card with comprehensive description"
        )
        print(f"Successfully updated README.md for {repo_id}")
        print(f"View at: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"Error updating README: {e}")

if __name__ == "__main__":
    main()