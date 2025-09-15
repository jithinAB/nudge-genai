# Quick Start Guide - LM Studio CSV Processor

## Features

✅ **Parallel Processing**: Processes multiple rows simultaneously with configurable concurrency
✅ **Rate Limiting**: Prevents API overload with configurable requests per minute
✅ **Crash Recovery**: Automatically saves checkpoints and can resume from where it left off
✅ **Retry Logic**: Automatic retries with exponential backoff for failed requests
✅ **Progress Tracking**: Real-time progress updates and detailed logging
✅ **Data Preservation**: Saves results incrementally to prevent data loss

## Installation

1. **Install Python dependencies:**
```bash
pip install aiohttp aiofiles
```

2. **Start LM Studio:**
   - Launch LM Studio
   - Load your preferred model
   - Start the server (default: http://localhost:1234)

## Configuration

Edit the script to adjust these settings:

```python
# API Configuration
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "local-model"  # Your model name

# Processing limits
MAX_CONCURRENT_REQUESTS = 3  # Parallel requests (increase for faster processing)
REQUESTS_PER_MINUTE = 20     # Rate limit (adjust based on your system)
MAX_RETRIES = 3              # Retry attempts for failed requests
REQUEST_TIMEOUT = 120        # Seconds before timeout
```

## Data Location

The CSV data file is located at: `../data/data/scenario.csv` (relative to scripts folder)
- Absolute path: `/home/bud/Desktop/nudge/data/data/scenario.csv`

## Row Number Identification

**Important:** Row numbers are 0-based, matching Python's indexing:
- Row 0 = First data row in CSV (after header)
- Row 1 = Second data row in CSV
- Row 99 = 100th data row in CSV

This makes it easy to:
- Locate specific rows in your CSV
- Access results by row number in code
- Track which personas succeeded/failed
- Resume processing from exact position

## Usage

### First Run
```bash
cd /home/bud/Desktop/nudge/scripts
python lm_studio_processor.py
```

The script will:
1. Test the connection to LM Studio
2. Start processing from the beginning
3. Save checkpoints every 5 rows
4. Create output in `synthetic_data_output/` folder

### Resume After Interruption

If the script crashes or is interrupted (Ctrl+C), simply run it again:

```bash
python lm_studio_processor.py
```

You'll see:
```
Found checkpoint from 2024-01-15 10:30:45
Progress: 45/100
Resume from checkpoint? (y/n): y
```

Type `y` to continue from where you left off.

## Output Files

The script creates several files in `synthetic_data_output/`:

1. **`synthetic_conversations_final.json`** - Final output indexed by row number
2. **`consolidated_results.json`** - All results (success & failures) indexed by row number
3. **`individual_results/`** - Folder with individual files per row:
   - `row_0000_User_01.json`
   - `row_0001_User_02.json`
   - etc.
4. **`failed_rows.json`** - Details of failed rows (if any)
5. **`processing_summary.json`** - Summary with row numbers of successful/failed
6. **`processing_[timestamp].log`** - Detailed processing log
7. **`processing_checkpoint.json`** - Checkpoint file (deleted after completion)

## Output Structure

Results are organized by row number. In the final JSON:
```json
{
  "0": {  // Row number 0
    "row_number": 0,
    "persona_name": "User 01",
    "success": true,
    "response": {
      "Conversations": {
        "scenario_name": [
          {
            "role": "person",
            "message": "..."
          },
          {
            "role": "AI",
            "message": "..."
          }
        ]
      },
      "metadata": {
        "row_number": 0,
        "persona_name": "User 01",
        "place": "Kerala",
        "demographics": "...",
        "beliefs": "...",
        "biases": "..."
      }
    },
    "timestamp": "2024-01-15T10:30:45"
  },
  "1": {  // Row number 1
    // Similar structure
  }
}
```

Individual files are named with row numbers for easy identification:
- `row_0000_User_01.json` - Row 0
- `row_0001_User_02.json` - Row 1
- etc.

## Monitoring Progress

The script provides real-time updates with row numbers:
```
2024-01-15 10:30:45 - INFO - Processing Row 0: User 01
2024-01-15 10:30:47 - INFO - Processing Row 1: User 02
2024-01-15 10:30:48 - INFO - Progress: 2/100 rows (2.0%)
```

The summary file shows exactly which rows succeeded or failed:
```json
{
  "total_rows": 100,
  "successful_rows": 98,
  "failed_rows": 2,
  "successful_row_numbers": [0, 1, 2, 3, ...],
  "failed_row_numbers": [45, 67]
}
```

## Troubleshooting

### Connection Failed
- Ensure LM Studio is running
- Check the model is loaded
- Verify server is on correct port

### Slow Processing
- Increase `MAX_CONCURRENT_REQUESTS` (e.g., to 5-10)
- Increase `REQUESTS_PER_MINUTE` if your system can handle it
- Reduce `MAX_TOKENS` if responses are too long

### Out of Memory
- Decrease `MAX_CONCURRENT_REQUESTS`
- The script processes in batches to manage memory

### Resume Not Working
- Check `processing_checkpoint.json` exists
- Ensure you're running from the same directory
- Don't modify the CSV file between runs

### CSV File Not Found
- Ensure the CSV file exists at `/home/bud/Desktop/nudge/data/data/scenario.csv`
- Check file permissions
- Run from the scripts directory

## Performance Tips

1. **For 100 rows with default settings:**
   - ~3 concurrent requests
   - ~20 requests/minute
   - Estimated time: 5-10 minutes

2. **For faster processing:**
   - Set `MAX_CONCURRENT_REQUESTS = 10`
   - Set `REQUESTS_PER_MINUTE = 60`
   - Estimated time: 2-3 minutes

3. **For stable processing:**
   - Keep defaults
   - Monitor system resources
   - Check LM Studio logs for errors

## Safety Features

- **Automatic checkpointing** every 5 rows
- **Graceful shutdown** on Ctrl+C (checkpoint saved)
- **Duplicate detection** prevents reprocessing
- **Incremental saves** prevent data loss
- **Error logging** for debugging

## CSV Data Structure

The script expects a CSV with these columns:
- `Persona Name` - Name/identifier for the persona
- `Place (Be specific)` - Location of the persona
- `Culture, Ethinicity, Demographics ` - Demographic information
- `Beliefs` - Beliefs of the persona
- `Biases` - Cognitive biases of the persona

## Example Commands

### Run the processor:
```bash
cd /home/bud/Desktop/nudge/scripts
python lm_studio_processor.py
```

### Check progress:
```bash
tail -f synthetic_data_output/processing_*.log
```

### View results:
```bash
ls synthetic_data_output/individual_results/
```

### Check summary:
```bash
cat synthetic_data_output/processing_summary.json
```