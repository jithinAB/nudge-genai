#!/usr/bin/env python3
"""
OpenAI CSV Processor for Bias-Based Nudging Conversations
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
from openai import AsyncOpenAI
import tiktoken
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ======================== CONFIGURATION ========================

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Set your OpenAI API key
MODEL_NAME = "gpt-5"  # Can be changed to gpt-4, gpt-3.5-turbo, etc.

# File paths
CSV_FILE = "../data/data/scenario.csv"
OUTPUT_DIR = "synthetic_data_output_openai"
CHECKPOINT_FILE = "processing_checkpoint.json"
FINAL_OUTPUT_FILE = "data_v2.json"
INDIVIDUAL_FILES = True

# Processing configuration
MAX_CONCURRENT_REQUESTS = 5  # OpenAI allows more concurrent requests
REQUESTS_PER_MINUTE = 30  # Adjust based on your OpenAI rate limits
MAX_RETRIES = 5
RETRY_DELAY_BASE = 2
REQUEST_TIMEOUT = 120  # 2 minutes timeout
SAVE_INTERVAL = 5

# Model parameters
TEMPERATURE = 0.8  # Higher temperature for more creative conversations
MAX_TOKENS = 4000  # Adjust based on expected response length

# ======================== PROMPT MANAGEMENT ========================

def load_prompt_template(prompt_name="s_to_convo_2.txt"):
    """Load prompt template from the prompts folder"""
    script_dir = Path(__file__).parent
    prompts_dir = script_dir.parent / "prompts"
    prompt_path = prompts_dir / prompt_name
    
    if not prompt_path.exists():
        print(f"Error: Prompt file not found: {prompt_path}")
        sys.exit(1)
        
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"Info: Loaded prompt template from {prompt_path}")
            return content
    except Exception as e:
        print(f"Error loading prompt template: {e}")
        sys.exit(1)

# Load the prompt template
PROMPT_TEMPLATE = load_prompt_template()

# ======================== DATA STRUCTURES ========================

@dataclass
class ProcessingResult:
    """Represents the result of processing a single row"""
    row_number: int
    persona_name: str
    success: bool
    response: Optional[Dict] = None
    reasoning: Optional[str] = None
    error: Optional[str] = None
    timestamp: str = ""
    retry_count: int = 0
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

@dataclass  
class CheckpointData:
    """Checkpoint data for resume capability"""
    processed_row_numbers: List[int]
    failed_row_numbers: List[int]
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

# ======================== TOKEN COUNTING ========================

def count_tokens(text: str, model: str = MODEL_NAME) -> int:
    """Count tokens in text for the specified model"""
    try:
        if "gpt-4" in model:
            encoding = tiktoken.encoding_for_model("gpt-4")
        elif "gpt-3.5" in model:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        else:
            encoding = tiktoken.get_encoding("cl100k_base")
        
        num_tokens = len(encoding.encode(text))
        return num_tokens
    except Exception as e:
        logger.warning(f"Token counting failed: {e}. Estimating...")
        # Rough estimate: 1 token ~= 4 characters
        return len(text) // 4

# ======================== API INTERACTION ========================

async def call_openai_api(
    client: AsyncOpenAI,
    prompt: str,
    semaphore: Semaphore,
    rate_limiter: RateLimiter,
    retry_count: int = 0
) -> tuple[bool, Optional[str], Optional[str], Optional[str]]:
    """
    Call OpenAI API with retry logic and rate limiting
    
    Returns: (success, response_text, reasoning, error_message)
    """
    async with semaphore:
        await rate_limiter.acquire()
        
        # Check token count
        prompt_tokens = count_tokens(prompt)
        if prompt_tokens > 8000:  # Leave room for response
            logger.warning(f"Prompt too long: {prompt_tokens} tokens. Truncating...")
            # Simple truncation - in production, use smarter truncation
            prompt = prompt[:30000]  # Roughly 7500 tokens
        
        messages = [
            {
                "role": "system",
                "content": "You are an expert psychologist simulating realistic conversations based on cognitive biases. Return your response as valid JSON only, without any markdown formatting or code blocks."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
        
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"}  # Force JSON response
            )
            
            content = response.choices[0].message.content
            
            # Try to extract reasoning if present in response
            try:
                parsed = json.loads(content)
                reasoning = parsed.get("analysis", {}).get("reasoning", None)
                return True, content, reasoning, None
            except:
                return True, content, None, None
                
        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower():
                logger.warning(f"Rate limit hit: {error_msg}")
                # Longer delay for rate limits
                await asyncio.sleep(60)
            elif "timeout" in error_msg.lower():
                logger.error(f"Request timeout")
            else:
                logger.error(f"API error: {error_msg}")
            
            return False, None, None, error_msg

def parse_json_response(response: str) -> Optional[Dict]:
    """Parse JSON from model response"""
    try:
        # OpenAI with json mode should return clean JSON
        parsed = json.loads(response)
        return parsed
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        logger.debug(f"Response: {response[:500]}...")
        
        # Save problematic response for debugging
        debug_file = Path(OUTPUT_DIR) / f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(debug_file, 'w') as f:
                f.write(response)
            logger.debug(f"Saved debug output to {debug_file}")
        except:
            pass
        
        return None

# ======================== ROW PROCESSING ========================

async def process_row(
    client: AsyncOpenAI,
    row: Dict[str, str],
    row_number: int,
    semaphore: Semaphore,
    rate_limiter: RateLimiter
) -> ProcessingResult:
    """Process a single CSV row"""
    
    persona_name = row.get('Persona Name', f'Row_{row_number}')
    
    # Extract data from CSV columns  
    demographics = row.get('Culture, Ethinicity, Demographics ', '')
    beliefs = row.get('Beliefs', '')
    biases = row.get('Biases', '').strip()
    
    # Clean up biases field
    if biases.endswith('.'):
        biases = biases[:-1].strip()
    
    # Format prompt using the template
    prompt = PROMPT_TEMPLATE.format(
        persona_number=f"Persona {row_number + 1:02d}",
        demographics=demographics,
        beliefs=beliefs, 
        biases=biases
    )
    
    logger.info(f"Processing Row {row_number}: {persona_name}")
    
    # Try with retries
    for retry in range(MAX_RETRIES):
        if retry > 0:
            delay = min(RETRY_DELAY_BASE * (2 ** (retry - 1)), 60)
            logger.warning(f"Retry {retry}/{MAX_RETRIES} for Row {row_number} after {delay}s delay")
            await asyncio.sleep(delay)
            
        success, response_text, reasoning, error = await call_openai_api(
            client, prompt, semaphore, rate_limiter, retry
        )
        
        if success and response_text:
            parsed = parse_json_response(response_text)
            if parsed:
                # Add metadata
                parsed['metadata'] = {
                    'row_number': row_number,
                    'persona_name': persona_name,
                    'demographics': demographics,
                    'beliefs': beliefs,
                    'biases': biases,
                    'timestamp': datetime.now().isoformat()
                }
                
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
    """Manages saving and loading results"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
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
        """Save individual result file"""
        if not result.success:
            return
            
        filename = self.individual_dir / f"row_{result.row_number:04d}_{result.persona_name.replace(' ', '_')}.json"
        
        output_data = {
            "row_number": result.row_number,
            "persona_name": result.persona_name,
            "timestamp": result.timestamp,
            "conversation": result.response.get('conversation', []) if result.response else [],
            "analysis": result.response.get('analysis', {}) if result.response else {},
            "metadata": result.response.get('metadata', {}) if result.response else {}
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(output_data, f, indent=2)
            logger.debug(f"Saved individual result for row {result.row_number}")
        except Exception as e:
            logger.error(f"Failed to save individual result for row {result.row_number}: {e}")
            
    async def save_consolidated(self):
        """Save all results in a consolidated file"""
        consolidated_file = self.output_dir / "consolidated_results.json"
        
        # Sort by row number
        sorted_results = dict(sorted(self.results.items()))
        
        try:
            with open(consolidated_file, 'w') as f:
                json.dump(sorted_results, f, indent=2)
            logger.debug(f"Saved {len(self.results)} consolidated results")
        except Exception as e:
            logger.error(f"Failed to save consolidated results: {e}")
            
    def get_successful_results(self) -> Dict[int, Dict]:
        """Extract only successful results"""
        return {
            row_num: data
            for row_num, data in self.results.items()
            if data['success']
        }
        
    def get_failed_results(self) -> Dict[int, Dict]:
        """Extract only failed results"""
        return {
            row_num: data
            for row_num, data in self.results.items()
            if not data['success']
        }

# ======================== MAIN PROCESSING ========================

async def process_csv(resume: bool = True):
    """Main processing function"""
    
    # Initialize OpenAI client
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY environment variable not set")
        sys.exit(1)
        
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
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
                    result_mgr.results = {int(k): v for k, v in existing_results.items()}
                    
    # Read CSV
    csv_path = Path(CSV_FILE)
    if not csv_path.is_absolute():
        csv_path = Path(__file__).parent / csv_path
        
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        sys.exit(1)
        
    logger.info(f"Reading CSV from: {csv_path}")
    
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    total_rows = len(rows)
    logger.info(f"Total rows in CSV: {total_rows}")
    logger.info(f"Rows to process: {total_rows - len(processed_row_numbers)}")
    
    # Initialize rate limiting
    semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)
    rate_limiter = RateLimiter(REQUESTS_PER_MINUTE, per=60.0)
    
    # Process rows
    tasks = []
    task_row_numbers = []
    
    for row_number, row in enumerate(rows):
        # Skip if already processed
        if row_number in processed_row_numbers:
            logger.info(f"Skipping already processed Row {row_number}")
            continue
            
        # Create processing task
        task = process_row(client, row, row_number, semaphore, rate_limiter)
        tasks.append(task)
        task_row_numbers.append(row_number)
        
        # Process in batches
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
    
    # Generate final output
    successful = result_mgr.get_successful_results()
    failed = result_mgr.get_failed_results()
    
    # Prepare final output
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
        
    # Save failed results
    if failed:
        failed_output_path = Path(OUTPUT_DIR) / "failed_rows.json"
        with open(failed_output_path, 'w') as f:
            json.dump(failed, f, indent=2)
            
    # Create summary
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
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
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
    """Test OpenAI connection before processing"""
    logger.info("Testing OpenAI connection...")
    
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not set")
        return False
        
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Say 'Connected' in JSON format"}],
            max_tokens=10,
            response_format={"type": "json_object"}
        )
        logger.info("✓ Connection successful!")
        return True
    except Exception as e:
        logger.error(f"✗ Connection failed: {e}")
        return False

async def main():
    """Main entry point"""
    # Set up signal handling
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("=" * 60)
    logger.info("OPENAI CSV PROCESSOR - Bias-Based Nudging")
    logger.info("=" * 60)
    logger.info(f"Model: {MODEL_NAME}")
    
    # Test connection
    if not await test_connection():
        logger.error("Please set OPENAI_API_KEY environment variable")
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