---
language:
- en
license: apache-2.0
size_categories:
- 1K<n<10K
task_categories:
- text-generation
- text-classification
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
    num_examples: 2500
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

- **100 Unique Personas**: Each with distinct demographic backgrounds and belief systems
- **25 Cognitive Biases**: Comprehensive coverage of major cognitive biases including:
  - Anchoring Bias
  - Authority Bias
  - Confirmation Bias
  - Framing Effect
  - Loss Aversion
  - And 20 more...
- **2500 Conversation Examples**: Each demonstrating a specific bias-persona combination
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

- **Total Conversations**: 2500
- **Unique Personas**: 100
- **Cognitive Biases Covered**: 25
- **Average Conversation Length**: ~13 turns
- **Generation Date**: 2025-10-10T19:57:19.561262
- **Model Used**: gpt-4o-mini

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{belief_biases_bench_2025,
  title={Belief and Biases Bench: A Synthetic Dataset for Cognitive Bias Detection},
  author={Shirin AP},
  year={2025},
  month={10},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/Shirinap123/Belief-and-Biases-Bench}
}
```

## License

This dataset is released under the Apache 2.0 License.

## Author

**Shirin AP**  
Email: iamshirinap@gmail.com

## Acknowledgments

This dataset was generated using OpenAI's GPT-4-mini model and structured to support research in cognitive bias detection and human-AI interaction.
