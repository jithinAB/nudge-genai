#!/usr/bin/env python3
"""
Robust CSV Processor for LM Studio API with Rate Limiting and Resume Capability
Features:
- Parallel processing with configurable rate limiting
- Automatic crash recovery and resume from last checkpoint
- Progress tracking and detailed logging
- Retry logic with exponential backoff
- Saves data with row number as identifier
"""

import csv
import json
import asyncio
import aiohttp
import aiofiles
import time
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from asyncio import Semaphore
import signal

# ======================== CONFIGURATION ========================

# API Configuration
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "openai/gpt-oss-20b"  # Updated to your loaded model name

# File paths - Updated to use correct data path
CSV_FILE = "../data/data/scenario.csv"  # Relative path from scripts folder
OUTPUT_DIR = "synthetic_data_output"
CHECKPOINT_FILE = "processing_checkpoint.json"
FINAL_OUTPUT_FILE = "data_v1.json"  # Main output file with all data including reasoning
INDIVIDUAL_FILES = True  # Save individual files for each row

# Processing configuration
MAX_CONCURRENT_REQUESTS = 1  # Sequential processing - one request at a time
REQUESTS_PER_MINUTE = 6  # Rate limit - 1 request every 10 seconds for stability
MAX_RETRIES = 5  # Increased retry attempts for better resilience
RETRY_DELAY_BASE = 3  # Base delay for exponential backoff (seconds)
REQUEST_TIMEOUT = 300  # 5 minutes timeout for complex reasoning
SAVE_INTERVAL = 1  # Save checkpoint after every processed row for safety

# Model parameters
TEMPERATURE = 0.7
MAX_TOKENS = 1500  # Optimized for shorter, complete JSON responses

# ======================== PROMPT TEMPLATE ========================

PROMPT_TEMPLATE = """Person Profile:
Location: {place}
Demographics: {demographics}
Beliefs: {beliefs}
Cognitive Biases: {bias}

Task: Create 2 realistic conversations between this person and an AI assistant.
Each conversation should have 3-4 exchanges and reflect the person's beliefs and biases.

Think step by step about how this person would interact based on their profile.

Return ONLY valid JSON in this format:
{{
  "Conversations": {{
    "scenario_1": [
      {{"role": "person", "message": "..."}},
      {{"role": "AI", "message": "..."}},
      {{"role": "person", "message": "..."}},
      {{"role": "AI", "message": "..."}}
    ],
    "scenario_2": [
      {{"role": "person", "message": "..."}},
      {{"role": "AI", "message": "..."}},
      {{"role": "person", "message": "..."}},
      {{"role": "AI", "message": "..."}}
    ]
  }}
}}"""

# ======================== DATA STRUCTURES ========================

@dataclass
class ProcessingResult:
    """Represents the result of processing a single row"""
    row_number: int  # Changed from row_id to row_number
    persona_name: str
    success: bool
    response: Optional[Dict] = None
    reasoning: Optional[str] = None  # To store reasoning if available
    error: Optional[str] = None
    timestamp: str = ""
    retry_count: int = 0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

@dataclass
class CheckpointData:
    """Checkpoint data for resume capability"""
    processed_row_numbers: List[int]  # Changed to store row numbers
    failed_row_numbers: List[int]     # Changed to store row numbers
    last_processed_row: int
    total_rows: int
    start_time: str
    last_update: str

# ======================== LOGGING SETUP ========================

def setup_logging():
    """Configure logging with both file and console output"""
    log_dir = Path(OUTPUT_DIR)
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ======================== CHECKPOINT MANAGEMENT ========================

class CheckpointManager:
    """Manages checkpoint saving and loading for crash recovery"""

    def __init__(self, checkpoint_file: str):
        self.checkpoint_file = Path(OUTPUT_DIR) / checkpoint_file
        self.checkpoint_file.parent.mkdir(exist_ok=True)

    def save(self, data: CheckpointData):
        """Save checkpoint data"""
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(asdict(data), f, indent=2)
            logger.debug(f"Checkpoint saved: {len(data.processed_row_numbers)} processed")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def load(self) -> Optional[CheckpointData]:
        """Load checkpoint data if exists"""
        if not self.checkpoint_file.exists():
            return None

        try:
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
            return CheckpointData(**data)
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def delete(self):
        """Delete checkpoint file after successful completion"""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logger.info("Checkpoint file deleted")

# ======================== RATE LIMITER ========================

class RateLimiter:
    """Token bucket rate limiter for API requests"""

    def __init__(self, rate: int, per: float = 60.0):
        self.rate = rate
        self.per = per
        self.allowance = rate
        self.last_check = time.monotonic()

    async def acquire(self):
        """Wait if necessary to respect rate limit"""
        current = time.monotonic()
        time_passed = current - self.last_check
        self.last_check = current
        self.allowance += time_passed * (self.rate / self.per)

        if self.allowance > self.rate:
            self.allowance = self.rate

        if self.allowance < 1.0:
            sleep_time = (1.0 - self.allowance) * (self.per / self.rate)
            logger.debug(f"Rate limit: sleeping for {sleep_time:.2f}s")
            await asyncio.sleep(sleep_time)
            self.allowance = 0.0
        else:
            self.allowance -= 1.0

# ======================== API INTERACTION ========================

async def call_lm_studio_api(
    session: aiohttp.ClientSession,
    prompt: str,
    semaphore: Semaphore,
    rate_limiter: RateLimiter,
    retry_count: int = 0
) -> tuple[bool, Optional[str], Optional[str], Optional[str]]:
    """
    Call LM Studio API with retry logic and rate limiting

    Returns: (success, response_text, reasoning, error_message)
    """
    async with semaphore:
        await rate_limiter.acquire()

        headers = {"Content-Type": "application/json"}
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert psychologist simulating realistic conversations. Think step by step about the person's profile before generating conversations. Return ONLY valid JSON without markdown or code blocks."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "stream": False
        }

        try:
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
            async with session.post(
                LM_STUDIO_URL,
                json=payload,
                headers=headers,
                timeout=timeout
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    message = result['choices'][0]['message']
                    content = message.get('content', '')

                    # Extract reasoning from the response if available
                    reasoning = message.get('reasoning', None)

                    # If reasoning is in the message itself, try to extract it
                    if not reasoning and 'reasoning' in message:
                        reasoning = message.get('reasoning')

                    # Log if we got reasoning
                    if reasoning:
                        logger.debug(f"Captured reasoning: {reasoning[:100]}...")

                    return True, content, reasoning, None
                else:
                    error = f"API returned status {response.status}"
                    return False, None, None, error

        except asyncio.TimeoutError:
            logger.error(f"Timeout after {REQUEST_TIMEOUT}s - model may be processing complex request")
            return False, None, None, f"Request timeout after {REQUEST_TIMEOUT}s"
        except aiohttp.ClientError as e:
            logger.error(f"Client error: {str(e)}")
            return False, None, None, f"Client error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return False, None, None, f"Unexpected error: {str(e)}"

def parse_json_response(response: str) -> Optional[Dict]:
    """Parse JSON from model response, handling markdown code blocks and various formats"""
    try:
        original_response = response

        # Remove markdown code blocks if present
        if "```json" in response.lower():
            start = response.lower().find("```json") + 7
            end = response.find("```", start)
            if end != -1:
                response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                response = response[start:end].strip()

        # Try to find JSON structure if not already clean
        if not response.strip().startswith('{'):
            # Look for JSON-like structure in the response
            json_start = response.find('{')
            if json_start != -1:
                response = response[json_start:]

        # Clean up common issues
        response = response.strip()

        # First attempt - try parsing as-is
        try:
            parsed = json.loads(response)
            if "Conversations" in parsed:
                return parsed
            else:
                logger.warning("JSON missing 'Conversations' key, wrapping response")
                return {"Conversations": parsed}
        except json.JSONDecodeError:
            pass

        # Second attempt - fix truncated JSON by counting and adding braces
        open_braces = response.count('{')
        close_braces = response.count('}')
        open_brackets = response.count('[')
        close_brackets = response.count(']')

        missing_braces = open_braces - close_braces
        missing_brackets = open_brackets - close_brackets

        if missing_braces > 0 or missing_brackets > 0:
            # Add missing brackets first, then braces
            fixed_response = response
            if missing_brackets > 0:
                fixed_response += ']' * missing_brackets
            if missing_braces > 0:
                fixed_response += '}' * missing_braces

            logger.warning(f"Fixed truncated JSON: added {missing_brackets} brackets and {missing_braces} braces")

            try:
                parsed = json.loads(fixed_response)
                if "Conversations" in parsed:
                    return parsed
                else:
                    return {"Conversations": parsed}
            except json.JSONDecodeError:
                pass

        # Third attempt - try to extract valid JSON portion
        # Find the last complete object/array
        stack = []
        last_valid_end = -1

        for i, char in enumerate(response):
            if char in '{[':
                stack.append((char, i))
            elif char in '}]':
                if stack:
                    open_char, _ = stack.pop()
                    if (char == '}' and open_char == '{') or (char == ']' and open_char == '['):
                        if not stack:  # Complete JSON structure
                            last_valid_end = i + 1

        if last_valid_end > 0:
            truncated = response[:last_valid_end]
            try:
                parsed = json.loads(truncated)
                logger.warning(f"Extracted valid JSON portion (truncated at position {last_valid_end})")
                if "Conversations" in parsed:
                    return parsed
                else:
                    return {"Conversations": parsed}
            except json.JSONDecodeError:
                pass

        # If all attempts fail, return None
        raise json.JSONDecodeError("Could not parse JSON after multiple attempts", response, 0)

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error after all attempts: {e}")
        logger.debug(f"Original response length: {len(original_response)}")
        logger.debug(f"Response preview: {original_response[:200]}...")

        # Save problematic response for debugging
        import hashlib
        debug_file = f"synthetic_data_output/debug_{hashlib.md5(original_response.encode()).hexdigest()[:8]}.txt"
        try:
            with open(debug_file, 'w') as f:
                f.write(f"Original:\n{original_response}\n\nCleaned:\n{response}")
            logger.debug(f"Saved debug output to {debug_file}")
        except:
            pass
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing JSON: {e}")
        return None

# ======================== ROW PROCESSING ========================

async def process_row(
    session: aiohttp.ClientSession,
    row: Dict[str, str],
    row_number: int,  # Using actual row number from CSV (0-based index)
    semaphore: Semaphore,
    rate_limiter: RateLimiter
) -> ProcessingResult:
    """Process a single CSV row"""

    persona_name = row.get('Persona Name', f'Row_{row_number}')

    # Extract data from CSV columns
    place = row.get('Place (Be specific)', '')
    demographics = row.get('Culture, Ethinicity, Demographics ', '')
    beliefs = row.get('Beliefs', '')
    biases = row.get('Biases', '')

    # Format prompt
    prompt = PROMPT_TEMPLATE.format(
        place=place,
        demographics=demographics,
        beliefs=beliefs,
        bias=biases
    )

    logger.info(f"Processing Row {row_number}: {persona_name}")

    # Try with retries and progressive backoff
    for retry in range(MAX_RETRIES):
        if retry > 0:
            delay = min(RETRY_DELAY_BASE * (2 ** (retry - 1)), 60)  # Cap at 60 seconds
            logger.warning(f"Retry {retry}/{MAX_RETRIES} for Row {row_number} after {delay}s delay")
            await asyncio.sleep(delay)

        success, response_text, reasoning, error = await call_lm_studio_api(
            session, prompt, semaphore, rate_limiter, retry
        )

        if success and response_text:
            parsed = parse_json_response(response_text)
            if parsed:
                # Add metadata with row number as primary identifier
                parsed['metadata'] = {
                    'row_number': row_number,
                    'persona_name': persona_name,
                    'place': place,
                    'demographics': demographics,
                    'beliefs': beliefs,
                    'biases': biases,
                    'timestamp': datetime.now().isoformat()
                }

                # Add reasoning if captured
                if reasoning:
                    parsed['reasoning'] = reasoning

                logger.info(f"✓ Successfully processed Row {row_number} on attempt {retry + 1}")
                return ProcessingResult(
                    row_number=row_number,
                    persona_name=persona_name,
                    success=True,
                    response=parsed,
                    reasoning=reasoning,
                    retry_count=retry
                )
        elif error:
            logger.warning(f"API error for Row {row_number}: {error}")

    # All retries failed
    logger.error(f"✗ Failed to process Row {row_number} after {MAX_RETRIES} attempts")
    return ProcessingResult(
        row_number=row_number,
        persona_name=persona_name,
        success=False,
        error=f"Failed after {MAX_RETRIES} retries. Last error: {error}",
        retry_count=MAX_RETRIES
    )

# ======================== RESULT MANAGEMENT ========================

class ResultManager:
    """Manages saving and loading results with row number as identifier"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}  # Store by row number
        self.individual_dir = self.output_dir / "individual_results"
        if INDIVIDUAL_FILES:
            self.individual_dir.mkdir(exist_ok=True)

    async def add_result(self, result: ProcessingResult):
        """Add a result indexed by row number"""
        self.results[result.row_number] = asdict(result)

        # Save individual file if enabled
        if INDIVIDUAL_FILES and result.success:
            await self.save_individual(result)

        # Save consolidated results periodically
        if len(self.results) % SAVE_INTERVAL == 0:
            await self.save_consolidated()

    async def save_individual(self, result: ProcessingResult):
        """Save individual result file named by row number"""
        if not result.success:
            return

        filename = self.individual_dir / f"row_{result.row_number:04d}_{result.persona_name.replace(' ', '_')}.json"

        output_data = {
            "row_number": result.row_number,
            "persona_name": result.persona_name,
            "timestamp": result.timestamp,
            "conversations": result.response.get('Conversations', {}) if result.response else {},
            "metadata": result.response.get('metadata', {}) if result.response else {},
            "reasoning": result.response.get('reasoning', result.reasoning) if result.response else result.reasoning
        }

        try:
            async with aiofiles.open(filename, 'w') as f:
                await f.write(json.dumps(output_data, indent=2))
            logger.debug(f"Saved individual result for row {result.row_number}")
        except Exception as e:
            logger.error(f"Failed to save individual result for row {result.row_number}: {e}")

    async def save_consolidated(self):
        """Save all results in a consolidated file organized by row number"""
        consolidated_file = self.output_dir / "consolidated_results.json"

        # Sort by row number for easy navigation
        sorted_results = dict(sorted(self.results.items()))

        try:
            async with aiofiles.open(consolidated_file, 'w') as f:
                await f.write(json.dumps(sorted_results, indent=2))
            logger.debug(f"Saved {len(self.results)} consolidated results")
        except Exception as e:
            logger.error(f"Failed to save consolidated results: {e}")

    def get_successful_results(self) -> Dict[int, Dict]:
        """Extract only successful results indexed by row number"""
        return {
            row_num: data
            for row_num, data in self.results.items()
            if data['success']
        }

    def get_failed_results(self) -> Dict[int, Dict]:
        """Extract only failed results indexed by row number"""
        return {
            row_num: data
            for row_num, data in self.results.items()
            if not data['success']
        }

# ======================== MAIN PROCESSING ========================

async def process_csv(resume: bool = True):
    """Main processing function"""

    # Initialize managers
    checkpoint_mgr = CheckpointManager(CHECKPOINT_FILE)
    result_mgr = ResultManager(OUTPUT_DIR)

    # Load checkpoint if resuming
    checkpoint = None
    processed_row_numbers = set()

    if resume:
        checkpoint = checkpoint_mgr.load()
        if checkpoint:
            logger.info(f"Resuming from checkpoint: Row {checkpoint.last_processed_row}/{checkpoint.total_rows}")
            processed_row_numbers = set(checkpoint.processed_row_numbers)

            # Load existing results
            consolidated_file = Path(OUTPUT_DIR) / "consolidated_results.json"
            if consolidated_file.exists():
                with open(consolidated_file, 'r') as f:
                    existing_results = json.load(f)
                    # Convert string keys back to integers
                    result_mgr.results = {int(k): v for k, v in existing_results.items()}

    # Read CSV - Handle both absolute and relative paths
    csv_path = Path(CSV_FILE)
    if not csv_path.is_absolute():
        # If relative, resolve from script directory
        csv_path = Path(__file__).parent / csv_path

    # Check if CSV file exists
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        logger.error(f"Looking in: {csv_path.absolute()}")
        sys.exit(1)

    logger.info(f"Reading CSV from: {csv_path}")

    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Process all rows - remove limit for production
    # rows = rows[:10]  # Uncomment to limit for testing

    total_rows = len(rows)
    logger.info(f"Total rows in CSV: {total_rows}")
    logger.info(f"Rows to process: {total_rows - len(processed_row_numbers)}")

    # Initialize rate limiting
    semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)
    rate_limiter = RateLimiter(REQUESTS_PER_MINUTE, per=60.0)

    # Process rows
    async with aiohttp.ClientSession() as session:
        tasks = []
        task_row_numbers = []

        for row_number, row in enumerate(rows):
            # Skip if already processed
            if row_number in processed_row_numbers:
                logger.info(f"Skipping already processed Row {row_number}")
                continue

            # Create processing task
            task = process_row(session, row, row_number, semaphore, rate_limiter)
            tasks.append(task)
            task_row_numbers.append(row_number)

            # Process immediately since MAX_CONCURRENT_REQUESTS = 1
            if len(tasks) >= MAX_CONCURRENT_REQUESTS:
                results = await asyncio.gather(*tasks)

                for result, row_num in zip(results, task_row_numbers):
                    await result_mgr.add_result(result)
                    processed_row_numbers.add(row_num)

                # Update checkpoint
                checkpoint_data = CheckpointData(
                    processed_row_numbers=list(processed_row_numbers),
                    failed_row_numbers=[r.row_number for r in results if not r.success],
                    last_processed_row=max(processed_row_numbers) if processed_row_numbers else 0,
                    total_rows=total_rows,
                    start_time=checkpoint.start_time if checkpoint else datetime.now().isoformat(),
                    last_update=datetime.now().isoformat()
                )
                checkpoint_mgr.save(checkpoint_data)

                tasks = []
                task_row_numbers = []

                # Progress update
                progress = (len(processed_row_numbers) / total_rows) * 100
                logger.info(f"Progress: {len(processed_row_numbers)}/{total_rows} rows ({progress:.1f}%)")

        # Process remaining tasks
        if tasks:
            results = await asyncio.gather(*tasks)
            for result, row_num in zip(results, task_row_numbers):
                await result_mgr.add_result(result)
                processed_row_numbers.add(row_num)

        # Final save
        await result_mgr.save_consolidated()

    # Generate final output organized by row number
    successful = result_mgr.get_successful_results()
    failed = result_mgr.get_failed_results()

    # Prepare final output with all data including reasoning
    final_data = {
        "metadata": {
            "total_rows": total_rows,
            "successful": len(successful),
            "failed": len(failed),
            "generation_time": datetime.now().isoformat(),
            "model": MODEL_NAME
        },
        "conversations": successful
    }

    # Save final successful results
    final_output_path = Path(OUTPUT_DIR) / FINAL_OUTPUT_FILE
    with open(final_output_path, 'w') as f:
        json.dump(final_data, f, indent=2)

    # Save failed results for review
    if failed:
        failed_output_path = Path(OUTPUT_DIR) / "failed_rows.json"
        with open(failed_output_path, 'w') as f:
            json.dump(failed, f, indent=2)

    # Create summary file
    summary_path = Path(OUTPUT_DIR) / "processing_summary.json"
    summary = {
        "total_rows": total_rows,
        "successful_rows": len(successful),
        "failed_rows": len(failed),
        "successful_row_numbers": sorted(successful.keys()),
        "failed_row_numbers": sorted(failed.keys()),
        "completion_time": datetime.now().isoformat()
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Summary
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info(f"Total rows: {total_rows}")
    logger.info(f"Successful: {len(successful)} (Rows: {sorted(successful.keys())[:10]}{'...' if len(successful) > 10 else ''})")
    logger.info(f"Failed: {len(failed)} (Rows: {sorted(failed.keys())[:10]}{'...' if len(failed) > 10 else ''})")
    logger.info(f"Final output: {final_output_path}")
    if INDIVIDUAL_FILES:
        logger.info(f"Individual files: {result_mgr.individual_dir}")
    logger.info(f"Summary: {summary_path}")

    # Clean up checkpoint
    checkpoint_mgr.delete()

# ======================== SIGNAL HANDLING ========================

def signal_handler(signum, frame):
    """Handle interruption gracefully"""
    logger.info("\nInterruption received. Checkpoint saved. Run script again to resume.")
    sys.exit(0)

# ======================== ENTRY POINT ========================

async def test_connection():
    """Test LM Studio connection before processing"""
    logger.info("Testing LM Studio connection...")

    test_prompt = "Respond with a simple JSON: {\"status\": \"connected\"}"

    async with aiohttp.ClientSession() as session:
        semaphore = Semaphore(1)
        rate_limiter = RateLimiter(60)

        success, response, reasoning, error = await call_lm_studio_api(
            session, test_prompt, semaphore, rate_limiter
        )

        if success:
            logger.info("✓ Connection successful!")
            return True
        else:
            logger.error(f"✗ Connection failed: {error}")
            logger.error("Please ensure:")
            logger.error("  1. LM Studio is running")
            logger.error("  2. A model is loaded")
            logger.error("  3. Server is started on http://localhost:1234")
            return False

async def main():
    """Main entry point"""
    # Set up signal handling for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("=" * 60)
    logger.info("LM STUDIO CSV PROCESSOR - Row Number Based")
    logger.info("=" * 60)

    # Test connection
    if not await test_connection():
        sys.exit(1)

    # Check for resume
    checkpoint_mgr = CheckpointManager(CHECKPOINT_FILE)
    checkpoint = checkpoint_mgr.load()

    if checkpoint:
        logger.info(f"Found checkpoint from {checkpoint.last_update}")
        logger.info(f"Last processed row: {checkpoint.last_processed_row}")
        logger.info(f"Total processed: {len(checkpoint.processed_row_numbers)} rows")
        response = input("Resume from checkpoint? (y/n): ").strip().lower()
        resume = response == 'y'
    else:
        resume = False

    # Start processing
    await process_csv(resume=resume)

if __name__ == "__main__":
    asyncio.run(main())