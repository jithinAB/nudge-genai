# Nudge GenAI - Synthetic Conversation Generator

A robust system for generating synthetic psychological conversations using Large Language Models (LLMs) via LM Studio. This project creates realistic multi-turn conversations based on demographic profiles, beliefs, and cognitive biases.

## 🎯 Overview

This project generates synthetic conversations between personas and AI assistants, where each persona is defined by:
- Geographic location
- Demographics (age, gender, education, etc.)
- Personal beliefs and values
- Cognitive biases

The system processes CSV data containing persona profiles and uses LLMs to generate contextually appropriate conversations that reflect the persona's characteristics.

## 🚀 Features

- **Parallel Processing**: Configurable concurrent request handling with rate limiting
- **Crash Recovery**: Automatic resume from last checkpoint after interruptions
- **Progress Tracking**: Detailed logging and real-time progress monitoring
- **Retry Logic**: Exponential backoff for failed requests
- **Flexible Output**: Individual JSON files per persona or consolidated output
- **Data Validation**: Robust JSON parsing with error handling
- **Rate Limiting**: Configurable requests per minute to prevent API overload

## 📋 Prerequisites

- Python 3.8+
- [LM Studio](https://lmstudio.ai/) installed and running
- A compatible LLM loaded in LM Studio (e.g., GPT-OSS-20B)

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone https://github.com/jithinAB/nudge-genai.git
cd nudge-genai
```

2. **Set up virtual environment**:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install aiohttp aiofiles
```

4. **Set up LM Studio**:
   - Download and install [LM Studio](https://lmstudio.ai/)
   - Load your preferred model (e.g., `openai/gpt-oss-20b`)
   - Start the local server (default: `http://localhost:1234`)

## 📁 Project Structure

```
nudge-genai/
├── data/
│   └── data/
│       └── scenario.csv          # Input CSV with persona profiles
├── scripts/
│   ├── lm_studio_processor.py    # Main processing script
│   ├── test_lm_studio.py        # LM Studio connection test
│   ├── test_simple.py           # Simple API test
│   └── synthetic_data_output/   # Generated conversations
│       ├── individual_results/  # Per-persona JSON files
│       ├── consolidated_results.json
│       ├── synthetic_conversations_final.json
│       ├── processing_summary.json
│       └── failed_rows.json
├── .gitignore
└── README.md
```

## 🔧 Configuration

Edit the configuration section in `scripts/lm_studio_processor.py`:

```python
# API Configuration
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "openai/gpt-oss-20b"

# Processing configuration
MAX_CONCURRENT_REQUESTS = 1  # Number of parallel requests
REQUESTS_PER_MINUTE = 10     # Rate limit
MAX_RETRIES = 3              # Maximum retry attempts
REQUEST_TIMEOUT = 180        # Timeout in seconds
SAVE_INTERVAL = 5            # Save checkpoint every N rows

# Model parameters
TEMPERATURE = 0.7
MAX_TOKENS = 2000
```

## 📊 Input Data Format

The CSV file should contain the following columns:
- `place`: Geographic location
- `demographics`: Age, gender, education, occupation, etc.
- `beliefs`: Personal beliefs and values
- `bias`: Cognitive biases

Example:
```csv
place,demographics,beliefs,bias
New York,"35, Male, MBA, Marketing Manager","Values work-life balance, Believes in sustainable living","Confirmation bias, Anchoring bias"
```

## 🚀 Usage

### 1. Test LM Studio Connection

```bash
cd scripts
python test_lm_studio.py
```

### 2. Run the Main Processor

**Start fresh processing**:
```bash
python lm_studio_processor.py
```

**Resume from checkpoint** (after interruption):
```bash
python lm_studio_processor.py --resume
```

### 3. Monitor Progress

The script provides real-time progress updates:
```
[INFO] Processing row 10/100 (10.0%) | Row ID: User_10
[INFO] Successfully processed row 10 in 3.45s
[INFO] Progress: 10/100 (10.0%) | Success rate: 90.0%
```

## 📤 Output Format

### Individual Result Files
Each persona generates a file in `synthetic_data_output/individual_results/`:

```json
{
  "row_number": 1,
  "row_id": "User_01",
  "status": "success",
  "processing_time": 3.45,
  "timestamp": "2025-01-16T10:30:00",
  "input_data": {
    "place": "New York",
    "demographics": "35, Male, MBA",
    "beliefs": "Values work-life balance",
    "bias": "Confirmation bias"
  },
  "output_data": {
    "Conversations": {
      "career_advice": [
        {"role": "person", "message": "..."},
        {"role": "AI", "message": "..."}
      ]
    }
  }
}
```

### Consolidated Output
`synthetic_conversations_final.json` contains all successful conversations in a single file.

### Processing Summary
`processing_summary.json` provides statistics:
```json
{
  "total_rows": 100,
  "successful": 95,
  "failed": 5,
  "success_rate": 95.0,
  "total_time": 450.5,
  "average_time_per_row": 4.5
}
```

## 🔍 Troubleshooting

### LM Studio Connection Issues
- Ensure LM Studio is running and the server is started
- Check the URL matches your LM Studio settings (default: `http://localhost:1234`)
- Verify the model name matches the loaded model

### Memory Issues
- Reduce `MAX_CONCURRENT_REQUESTS` to 1
- Decrease `MAX_TOKENS` if responses are too large
- Process data in smaller batches

### JSON Parsing Errors
- Check debug files in `synthetic_data_output/debug_*.txt`
- Review the prompt template to ensure it requests valid JSON
- Increase `MAX_TOKENS` if responses are being truncated

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [LM Studio](https://lmstudio.ai/) for local LLM inference
- Uses OpenAI-compatible API for maximum flexibility
- Inspired by research in synthetic data generation for AI training

## 📧 Contact

For questions or support, please open an issue on GitHub or contact the maintainers.

---

**Note**: This tool is designed for research and development purposes. Ensure you comply with all applicable data protection and privacy regulations when generating synthetic data based on real demographic profiles.