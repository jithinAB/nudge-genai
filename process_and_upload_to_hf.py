#!/usr/bin/env python3
"""
Process and upload Belief and Biases dataset to HuggingFace
"""

import json
import os
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, create_repo, upload_folder
import pandas as pd
from datetime import datetime

# Configuration
HF_TOKEN = os.environ.get("HF_TOKEN")  # Set this environment variable with your HuggingFace token
DATASET_NAME = "Belief-and-Biases-Bench"
INPUT_FILE = "nudge-genai/scripts/synthetic_data_cross_bias_output/cross_bias_conversations.json"

if not HF_TOKEN:
    print("Error: Please set the HF_TOKEN environment variable with your HuggingFace token")
    print("Example: export HF_TOKEN='your_token_here'")
    exit(1)

def load_and_process_data(file_path):
    """Load and process the JSON data into a format suitable for HuggingFace datasets"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract metadata
    metadata = data['metadata']
    results = data['results_by_user']
    
    # Prepare dataset rows
    dataset_rows = []
    
    for user_id, user_data in results.items():
        for bias_name, bias_data in user_data.items():
            if bias_data['success'] and bias_data['response']:
                response = bias_data['response']
                
                # Create a row for each conversation
                row = {
                    'user_id': int(user_id),
                    'persona_name': response['metadata']['persona_name'],
                    'demographics': response['metadata']['demographics'],
                    'beliefs': response['metadata']['beliefs'],
                    'original_bias': response['metadata']['original_bias'],
                    'tested_bias': response['metadata']['tested_bias'],
                    'conversation': response['conversation'],
                    'utilised_bias_name': response['analysis']['utilised_bias']['name'],
                    'utilised_bias_method': response['analysis']['utilised_bias']['method'],
                    'nudging_bias_name': response['analysis']['nudging_bias']['name'],
                    'nudging_bias_method': response['analysis']['nudging_bias']['method'],
                    'reasoning_steps': response['analysis']['reasoning'],
                    'timestamp': response['metadata']['timestamp']
                }
                dataset_rows.append(row)
    
    return dataset_rows, metadata

def create_dataset_card(metadata):
    """Create a comprehensive dataset card for HuggingFace"""
    card_content = f"""---
language:
- en
license: apache-2.0
size_categories:
- 1K<n<10K
task_categories:
- text-generation
- text-classification
- conversational
pretty_name: Belief and Biases Bench
tags:
- cognitive-biases
- behavioral-economics
- conversation-analysis
- belief-systems
- synthetic-data
- bias-detection
- human-behavior
- psychology
dataset_info:
  features:
  - name: user_id
    dtype: int64
  - name: persona_name
    dtype: string
  - name: demographics
    dtype: string
  - name: beliefs
    dtype: string
  - name: original_bias
    dtype: string
  - name: tested_bias
    dtype: string
  - name: conversation
    list:
      - name: role
        dtype: string
      - name: message
        dtype: string
  - name: utilised_bias_name
    dtype: string
  - name: utilised_bias_method
    dtype: string
  - name: nudging_bias_name
    dtype: string
  - name: nudging_bias_method
    dtype: string
  - name: reasoning_steps
    sequence: string
  - name: timestamp
    dtype: string
  splits:
  - name: train
    num_examples: {metadata['successful_combinations']}
---

# Belief and Biases Bench

## Dataset Description

The **Belief and Biases Bench** is a comprehensive synthetic dataset designed to evaluate the capability of Large Language Models (LLMs) in accurately identifying beliefs and cognitive biases in human conversations. This dataset provides a rich collection of conversational interactions where various cognitive biases are exhibited and addressed through strategic nudging techniques.

### Purpose and Applications

This dataset is specifically created to:

1. **Evaluate LLM Bias Detection**: Test how well language models can identify cognitive biases in human communication
2. **Study Nudging Techniques**: Analyze effective strategies for addressing and redirecting cognitive biases
3. **Train Bias-Aware Models**: Develop AI systems that can recognize and appropriately respond to human cognitive biases
4. **Research Human-AI Interaction**: Study how AI can better understand and adapt to human belief systems and biases
5. **Improve Conversational AI**: Enhance chatbots and virtual assistants to be more aware of human cognitive patterns

### Key Features

- **{metadata['total_users']} Unique Personas**: Each with distinct demographic backgrounds and belief systems
- **{len(metadata['unique_biases'])} Cognitive Biases**: Comprehensive coverage of major cognitive biases including:
  - Anchoring Bias
  - Authority Bias
  - Confirmation Bias
  - Framing Effect
  - Loss Aversion
  - And {len(metadata['unique_biases']) - 5} more...
- **{metadata['successful_combinations']} Conversation Examples**: Each demonstrating a specific bias-persona combination
- **Detailed Analysis**: Each conversation includes:
  - The cognitive bias being exhibited
  - The nudging technique employed
  - Step-by-step reasoning of the approach
  - Metadata about the persona and context

## Dataset Structure

Each entry in the dataset contains:

### Main Fields

- `user_id`: Unique identifier for the persona (0-99)
- `persona_name`: Name of the synthetic persona (e.g., "User 01")
- `demographics`: Detailed demographic information including age, background, education
- `beliefs`: Core beliefs and values of the persona
- `original_bias`: The primary bias tendency of the persona
- `tested_bias`: The specific cognitive bias being demonstrated in the conversation

### Conversation Data

- `conversation`: A list of message exchanges between user and assistant
  - `role`: Either "user" or "assistant"
  - `message`: The actual message content

### Analysis Fields

- `utilised_bias_name`: The cognitive bias exhibited by the user
- `utilised_bias_method`: Description of how the bias manifests
- `nudging_bias_name`: The technique used to address the bias
- `nudging_bias_method`: Explanation of the nudging approach
- `reasoning_steps`: Detailed breakdown of the conversational strategy

## Use Cases

### 1. Bias Detection Research
Researchers can use this dataset to develop and test algorithms for detecting cognitive biases in text conversations.

### 2. Conversational AI Training
Train chatbots and virtual assistants to recognize and appropriately respond to various cognitive biases.

### 3. Behavioral Economics Studies
Analyze how different demographic groups exhibit various cognitive biases and how they respond to different nudging strategies.

### 4. Educational Tools
Develop educational applications that help people recognize their own cognitive biases through interactive examples.

### 5. Mental Health Applications
Create supportive AI systems that can identify potentially harmful thought patterns and provide gentle redirection.

## Ethical Considerations

- This is a synthetic dataset created using GPT-4-mini
- Personas and conversations are fictional and do not represent real individuals
- The dataset should be used responsibly and not to manipulate or exploit cognitive biases
- Users should be aware that bias detection and nudging techniques should be applied ethically

## Dataset Statistics

- **Total Conversations**: {metadata['successful_combinations']}
- **Unique Personas**: {metadata['total_users']}
- **Cognitive Biases Covered**: {len(metadata['unique_biases'])}
- **Average Conversation Length**: ~13 turns
- **Generation Date**: {metadata['generation_time']}
- **Model Used**: {metadata['model']}

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{{belief_biases_bench_2025,
  title={{Belief and Biases Bench: A Synthetic Dataset for Cognitive Bias Detection}},
  author={{Anonymous}},
  year={{2025}},
  month={{10}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/datasets/[username]/Belief-and-Biases-Bench}}
}}
```

## License

This dataset is released under the Apache 2.0 License.

## Acknowledgments

This dataset was generated using OpenAI's GPT-4-mini model and structured to support research in cognitive bias detection and human-AI interaction.
"""
    
    return card_content

def main():
    print("Loading dataset...")
    dataset_rows, metadata = load_and_process_data(INPUT_FILE)
    
    print(f"Loaded {len(dataset_rows)} conversations")
    
    # Create HuggingFace dataset
    dataset = Dataset.from_list(dataset_rows)
    
    # Create dataset dict with train split
    dataset_dict = DatasetDict({
        "train": dataset
    })
    
    # Create output directory
    output_dir = "hf_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save dataset locally
    dataset_dict.save_to_disk(output_dir)
    
    # Create and save dataset card
    dataset_card = create_dataset_card(metadata)
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(dataset_card)
    
    print("Dataset prepared locally. Now uploading to HuggingFace...")
    
    # Initialize HuggingFace API
    api = HfApi(token=HF_TOKEN)
    
    try:
        # Create repository
        repo_url = create_repo(
            repo_id=DATASET_NAME,
            token=HF_TOKEN,
            private=False,
            repo_type="dataset"
        )
        print(f"Created repository: {repo_url}")
    except Exception as e:
        print(f"Repository might already exist or error creating: {e}")
        repo_url = f"https://huggingface.co/datasets/{api.whoami()['name']}/{DATASET_NAME}"
    
    # Push dataset to hub
    dataset_dict.push_to_hub(
        DATASET_NAME,
        token=HF_TOKEN,
        private=False
    )
    
    print(f"Dataset successfully uploaded to: {repo_url}")
    print("Done!")

if __name__ == "__main__":
    main()