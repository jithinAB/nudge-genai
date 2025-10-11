# HuggingFace Dataset Upload Scripts

These scripts process and upload the Belief and Biases dataset to HuggingFace.

## Setup

1. Set your HuggingFace token as an environment variable:
   ```bash
   export HF_TOKEN='your_huggingface_token_here'
   ```

2. Install required dependencies:
   ```bash
   pip install datasets huggingface_hub
   ```

## Usage

### Processing and Uploading Dataset

Run the main processing script:
```bash
python process_and_upload_to_hf.py
```

This script will:
- Load the synthetic bias conversations from `nudge-genai/scripts/synthetic_data_cross_bias_output/cross_bias_conversations.json`
- Process the data into HuggingFace dataset format
- Create a new dataset repository named "Belief-and-Biases-Bench"
- Upload the dataset with a comprehensive dataset card

### Updating Dataset Card

If you need to update just the README/dataset card:
```bash
python update_hf_readme.py
```

## Dataset Information

- **Name**: Belief-and-Biases-Bench
- **Author**: Shirin AP (iamshirinap@gmail.com)
- **Size**: 2,500 conversations
- **Coverage**: 25 cognitive biases across 100 unique personas
- **Purpose**: Evaluate LLM capabilities in identifying beliefs and cognitive biases

## Security Note

Never commit your HuggingFace token to version control. Always use environment variables for sensitive credentials.