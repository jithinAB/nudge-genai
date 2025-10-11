# Cross-Bias Processing Status

## Current Run Configuration
- **Users**: 10 (User 01 to User 10)
- **Biases**: 25 (All unique cognitive biases)
- **Total Combinations**: 250 conversations
- **Started**: 2025-10-10 04:10:06

## Processing Details
- **Process ID**: 73535
- **Log File**: cross_bias_processing.log
- **Output Directory**: synthetic_data_cross_bias_output/
- **Running in**: Background (nohup)

## How to Monitor

### Live Progress
```bash
tail -f cross_bias_processing.log
```

### Check Summary
```bash
# Count successful completions
grep -c "✓ Success:" cross_bias_processing.log

# Check current progress
grep "Progress:" cross_bias_processing.log | tail -1

# See latest activity
tail -20 cross_bias_processing.log
```

### Check if Still Running
```bash
ps aux | grep openai_cross_bias_processor.py
```

## Expected Output Structure
```
synthetic_data_cross_bias_output/
├── cross_bias_conversations.json          # Main output
├── consolidated_cross_bias_results.json   # All results
├── individual_results/
│   ├── user_000_User_01/
│   │   ├── anchoring_bias.json
│   │   ├── authority_bias.json
│   │   └── ... (25 bias files)
│   ├── user_001_User_02/
│   │   └── ... (25 bias files)
│   └── ... (10 user directories total)
```

## Estimated Completion Time
- Rate: ~30 requests per minute (rate limited)
- Expected duration: ~8-10 minutes for 250 conversations
- With retries and delays: ~15-20 minutes total

## Resume Capability
If the process crashes or is interrupted:
1. The checkpoint file `cross_bias_checkpoint.json` saves progress
2. Rerun the script and it will prompt to resume
3. Already completed combinations will be skipped

## Final Output
When complete, check:
- `cross_bias_conversations.json` for the main results
- `processing_summary.json` for statistics
- `failed_rows.json` (if any failures occur)