# Cross-Bias Processor

This script generates conversations for each user persona with every unique cognitive bias, creating a comprehensive dataset for bias-based nudging analysis.

## Overview

The cross-bias processor:
- Extracts 25 unique cognitive biases from the dataset
- Generates conversations for each of the 100 users with each of the 25 biases
- Creates 2,500 total user-bias combinations
- Organizes output by user ID and bias type

## Key Features

1. **Comprehensive Coverage**: Tests every user with every cognitive bias
2. **Organized Output**: Results structured by user ID → bias type
3. **Resume Capability**: Can resume processing if interrupted
4. **Rate Limiting**: Respects OpenAI API rate limits
5. **Individual Files**: Optionally saves each result separately

## Usage

```bash
# Make sure you have the API key set
export OPENAI_API_KEY="your-key-here"
# Or use the .env file

# Install dependencies
pip install -r requirements.txt

# Run the processor
cd scripts
python openai_cross_bias_processor.py
```

## Output Structure

```
synthetic_data_cross_bias_output/
├── cross_bias_conversations.json       # Main output file
├── consolidated_cross_bias_results.json # All results by user
├── cross_bias_checkpoint.json          # Resume checkpoint
├── processing_summary.json             # Processing statistics
└── individual_results/                 # Optional individual files
    ├── user_000_User_01/
    │   ├── anchoring_bias.json
    │   ├── authority_bias.json
    │   └── ... (one file per bias)
    └── user_001_User_02/
        └── ... (25 bias files)
```

## Main Output Format

```json
{
  "metadata": {
    "total_users": 100,
    "total_biases": 25,
    "total_combinations": 2500,
    "successful_combinations": 2500,
    "generation_time": "2024-01-20T10:30:00",
    "model": "gpt-4o-mini",
    "unique_biases": ["Anchoring Bias", "Authority Bias", ...]
  },
  "results_by_user": {
    "0": {  // User row number
      "Anchoring Bias": {
        "conversation": [...],
        "analysis": {...},
        "metadata": {...}
      },
      "Authority Bias": {...},
      // ... all 25 biases
    },
    "1": {...},
    // ... all 100 users
  }
}
```

## Configuration

Edit the script to modify:
- `MAX_CONCURRENT_REQUESTS`: Number of parallel API calls (default: 5)
- `REQUESTS_PER_MINUTE`: API rate limit (default: 30)
- `MODEL_NAME`: OpenAI model to use (default: "gpt-4o-mini")
- `INDIVIDUAL_FILES`: Save individual files per user-bias (default: True)

## Unique Biases Tested

All 25 cognitive biases tested for each user:
1. Anchoring Bias
2. Authority Bias
3. Availability Bias
4. Availability Heuristic
5. Bandwagon Effect
6. Belief Bias
7. Confirmation Bias
8. Endowment Effect
9. Framing Effect
10. Gambler's Fallacy
11. Halo Effect
12. Hindsight Bias
13. Hyperbolic Discounting
14. IKEA Effect
15. In-group Bias
16. Loss Aversion
17. Negativity Bias
18. Optimism Bias
19. Peak-End Rule
20. Present Bias
21. Risk Aversion
22. Scarcity Bias
23. Social Proof
24. Status Quo Bias
25. Sunk Cost Fallacy

## Processing Time

With default settings:
- ~2,500 API calls required
- At 30 requests/minute: ~83 minutes minimum
- Actual time depends on API response times and rate limits