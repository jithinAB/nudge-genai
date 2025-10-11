#!/usr/bin/env python3
"""
Convert the Belief and Biases dataset to CSV format
"""

import json
import pandas as pd
import os
from datasets import load_dataset

def convert_conversations_to_string(conversations):
    """Convert conversation list to a formatted string"""
    conv_str = ""
    for turn in conversations:
        role = turn['role'].upper()
        message = turn['message']
        conv_str += f"{role}: {message}\n\n"
    return conv_str.strip()

def convert_json_to_csv(input_file, output_file):
    """Convert the JSON dataset to CSV format"""
    print(f"Loading data from {input_file}...")
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    results = data['results_by_user']
    
    # Prepare rows for CSV
    csv_rows = []
    
    for user_id, user_data in results.items():
        for bias_name, bias_data in user_data.items():
            if bias_data['success'] and bias_data['response']:
                response = bias_data['response']
                
                # Flatten the conversation into a readable format
                conversation_text = convert_conversations_to_string(response['conversation'])
                
                # Join reasoning steps with numbered list
                reasoning_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(response['analysis']['reasoning'])])
                
                row = {
                    'user_id': int(user_id),
                    'persona_name': response['metadata']['persona_name'],
                    'demographics': response['metadata']['demographics'],
                    'beliefs': response['metadata']['beliefs'],
                    'original_bias': response['metadata']['original_bias'],
                    'tested_bias': response['metadata']['tested_bias'],
                    'conversation': conversation_text,
                    'utilised_bias_name': response['analysis']['utilised_bias']['name'],
                    'utilised_bias_method': response['analysis']['utilised_bias']['method'],
                    'nudging_bias_name': response['analysis']['nudging_bias']['name'],
                    'nudging_bias_method': response['analysis']['nudging_bias']['method'],
                    'reasoning_steps': reasoning_text,
                    'timestamp': response['metadata']['timestamp']
                }
                csv_rows.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(csv_rows)
    
    # Sort by user_id and tested_bias for better organization
    df = df.sort_values(['user_id', 'tested_bias'])
    
    print(f"Saving {len(df)} rows to {output_file}...")
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"CSV file saved successfully!")
    
    return df

def convert_hf_dataset_to_csv(dataset_name, output_file):
    """Load dataset from HuggingFace and convert to CSV"""
    print(f"Loading dataset from HuggingFace: {dataset_name}...")
    
    # Load the dataset
    dataset = load_dataset(dataset_name)
    
    # Get the train split
    train_data = dataset['train']
    
    # Convert to pandas DataFrame
    df = train_data.to_pandas()
    
    # Process conversation column to make it readable
    df['conversation'] = df['conversation'].apply(convert_conversations_to_string)
    
    # Process reasoning_steps to make it readable
    df['reasoning_steps'] = df['reasoning_steps'].apply(lambda x: "\n".join([f"{i+1}. {step}" for i, step in enumerate(x)]))
    
    # Save to CSV
    print(f"Saving {len(df)} rows to {output_file}...")
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"CSV file saved successfully!")
    
    return df

def create_summary_csv(df, output_file):
    """Create a summary CSV with key statistics"""
    summary_data = []
    
    # Get unique biases and personas
    unique_biases = df['tested_bias'].unique()
    unique_personas = df['persona_name'].unique()
    
    for bias in sorted(unique_biases):
        bias_df = df[df['tested_bias'] == bias]
        summary_data.append({
            'bias_type': bias,
            'total_conversations': len(bias_df),
            'unique_personas': len(bias_df['persona_name'].unique()),
            'example_nudging_technique': bias_df.iloc[0]['nudging_bias_name'] if len(bias_df) > 0 else 'N/A'
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_file, index=False)
    print(f"Summary CSV saved to {output_file}")
    
    return summary_df

def main():
    # File paths
    json_input = "nudge-genai/scripts/synthetic_data_cross_bias_output/cross_bias_conversations.json"
    csv_output = "belief_and_biases_dataset.csv"
    summary_output = "belief_and_biases_summary.csv"
    
    # Option 1: Convert from local JSON file
    if os.path.exists(json_input):
        print("Converting from local JSON file...")
        df = convert_json_to_csv(json_input, csv_output)
    else:
        # Option 2: Download from HuggingFace and convert
        print("Local file not found. Downloading from HuggingFace...")
        df = convert_hf_dataset_to_csv("Shirinap123/Belief-and-Biases-Bench", csv_output)
    
    # Create summary CSV
    create_summary_csv(df, summary_output)
    
    # Print some statistics
    print("\nDataset Statistics:")
    print(f"Total conversations: {len(df)}")
    print(f"Unique personas: {df['persona_name'].nunique()}")
    print(f"Unique biases tested: {df['tested_bias'].nunique()}")
    print(f"\nFiles created:")
    print(f"- {csv_output} (full dataset)")
    print(f"- {summary_output} (summary statistics)")

if __name__ == "__main__":
    main()