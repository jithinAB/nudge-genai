#!/usr/bin/env python3
"""
OpenAI Cross-Bias Processor for Nudging Conversations
Generates conversations for each user with every unique cognitive bias
Features:
- Extracts unique biases from the dataset
- Creates user-bias combinations for comprehensive coverage
- Parallel processing with rate limiting
- Resume capability and checkpoint management
- Organized output by user ID and bias type
"""

import csv
import json
import asyncio
import time
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from asyncio import Semaphore
import signal
from openai import AsyncOpenAI
import tiktoken
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# ======================== CONFIGURATION ========================

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o-mini"

# File paths
CSV_FILE = "../data/data/scenario.csv"
OUTPUT_DIR = "synthetic_data_cross_bias_output"
CHECKPOINT_FILE = "cross_bias_checkpoint.json"
FINAL_OUTPUT_FILE = "cross_bias_conversations.json"
INDIVIDUAL_FILES = True

# Processing configuration
MAX_CONCURRENT_REQUESTS = 5
REQUESTS_PER_MINUTE = 30
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2
REQUEST_TIMEOUT = 120
SAVE_INTERVAL = 10

# Model parameters
TEMPERATURE = 0.8
MAX_TOKENS = 4000

# ======================== COGNITIVE BIASES ========================

# TEST MODE: Limited biases for testing
TEST_MODE = True
TEST_ROWS = 2
TEST_BIASES = 2

# Comprehensive list of all unique biases (cleaned and standardized)
ALL_UNIQUE_BIASES = [
    "Anchoring Bias",
    "Authority Bias", 
    "Availability Bias",
    "Availability Heuristic",
    "Bandwagon Effect",
    "Belief Bias",
    "Confirmation Bias",
    "Endowment Effect",
    "Framing Effect",
    "Gambler's Fallacy",
    "Halo Effect",
    "Hindsight Bias",
    "Hyperbolic Discounting",
    "IKEA Effect",
    "In-group Bias",
    "Loss Aversion",
    "Negativity Bias",
    "Optimism Bias",
    "Peak-End Rule",
    "Present Bias",
    "Risk Aversion",
    "Scarcity Bias",
    "Social Proof",
    "Status Quo Bias",
    "Sunk Cost Fallacy"
]

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

PROMPT_TEMPLATE = load_prompt_template()

# ======================== DATA STRUCTURES ========================

@dataclass
class UserBiasProcessingResult:
    """Result of processing a user-bias combination"""
    row_number: int
    persona_name: str
    original_bias: str
    tested_bias: str
    success: bool
    response: Optional[Dict] = None
    error: Optional[str] = None
    timestamp: str = ""
    retry_count: int = 0
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

@dataclass
class CrossBiasCheckpoint:
    """Checkpoint for cross-bias processing"""
    processed_combinations: List[Tuple[int, str]]  # (row_number, bias)
    failed_combinations: List[Tuple[int, str]]
    total_combinations: int
    total_users: int
    total_biases: int
    start_time: str
    last_update: str

# ======================== LOGGING SETUP ========================

def setup_logging():
    """Configure logging"""
    log_dir = Path(OUTPUT_DIR)
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"cross_bias_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
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

# ======================== BIAS UTILITIES ========================

def clean_bias_name(bias: str) -> str:
    """Clean and standardize bias names"""
    # Remove trailing punctuation and extra spaces
    bias = re.sub(r'[.,\s]+$', '', bias.strip())
    
    # Fix common typos
    typo_fixes = {
        'suck cost': 'sunk cost',
        'conformation': 'confirmation',
        'bandwagon effec': 'bandwagon effect',
        'availability heuristics': 'availability heuristic',
        'scarcity effect': 'scarcity bias'
    }
    
    bias_lower = bias.lower()
    for typo, fix in typo_fixes.items():
        if typo in bias_lower:
            bias_lower = bias_lower.replace(typo, fix)
    
    # Convert to title case
    words = bias_lower.split()
    return ' '.join(word.capitalize() for word in words)

def extract_unique_biases(csv_path: Path) -> Set[str]:
    """Extract and clean unique biases from CSV"""
    biases = set()
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            bias = row.get('Biases', '').strip()
            if bias:
                cleaned = clean_bias_name(bias)
                if cleaned:
                    biases.add(cleaned)
    
    return biases

# ======================== CHECKPOINT MANAGEMENT ========================

class CrossBiasCheckpointManager:
    """Manages checkpoints for cross-bias processing"""
    
    def __init__(self, checkpoint_file: str):
        self.checkpoint_file = Path(OUTPUT_DIR) / checkpoint_file
        self.checkpoint_file.parent.mkdir(exist_ok=True)
        
    def save(self, data: CrossBiasCheckpoint):
        """Save checkpoint"""
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(asdict(data), f, indent=2)
            logger.debug(f"Checkpoint saved: {len(data.processed_combinations)} processed")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            
    def load(self) -> Optional[CrossBiasCheckpoint]:
        """Load checkpoint"""
        if not self.checkpoint_file.exists():
            return None
            
        try:
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
            # Convert lists back to tuples
            data['processed_combinations'] = [tuple(x) for x in data['processed_combinations']]
            data['failed_combinations'] = [tuple(x) for x in data['failed_combinations']]
            return CrossBiasCheckpoint(**data)
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
            
    def delete(self):
        """Delete checkpoint file"""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logger.info("Checkpoint file deleted")

# ======================== RATE LIMITER ========================

class RateLimiter:
    """Token bucket rate limiter"""
    
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

async def call_openai_api(
    client: AsyncOpenAI,
    prompt: str,
    semaphore: Semaphore,
    rate_limiter: RateLimiter
) -> tuple[bool, Optional[str], Optional[str]]:
    """Call OpenAI API with rate limiting"""
    async with semaphore:
        await rate_limiter.acquire()
        
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
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            return True, content, None
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"API error: {error_msg}")
            return False, None, error_msg

def clean_unicode_quotes(text: str) -> str:
    """Replace Unicode quotes with standard ASCII quotes"""
    replacements = {
        '\u2018': "'",  # Left single quote
        '\u2019': "'",  # Right single quote
        '\u201c': '"',  # Left double quote
        '\u201d': '"',  # Right double quote
        '\u2013': '-',  # En dash
        '\u2014': '--', # Em dash
        '\u2026': '...',  # Ellipsis
    }
    for unicode_char, ascii_char in replacements.items():
        text = text.replace(unicode_char, ascii_char)
    return text

def clean_response_dict(data: Any) -> Any:
    """Recursively clean Unicode characters from dictionary"""
    if isinstance(data, dict):
        return {k: clean_response_dict(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_response_dict(item) for item in data]
    elif isinstance(data, str):
        return clean_unicode_quotes(data)
    else:
        return data

def parse_json_response(response: str) -> Optional[Dict]:
    """Parse JSON from model response and clean Unicode characters"""
    try:
        parsed = json.loads(response)
        # Clean Unicode characters from the parsed data
        cleaned = clean_response_dict(parsed)
        return cleaned
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return None

# ======================== PROCESSING ========================

async def process_user_bias_combination(
    client: AsyncOpenAI,
    row: Dict[str, str],
    row_number: int,
    test_bias: str,
    semaphore: Semaphore,
    rate_limiter: RateLimiter
) -> UserBiasProcessingResult:
    """Process a single user-bias combination"""
    
    persona_name = row.get('Persona Name', f'User_{row_number:02d}')
    original_bias = row.get('Biases', '')
    
    # Extract user data
    demographics = row.get('Culture, Ethinicity, Demographics ', '')
    beliefs = row.get('Beliefs', '')
    
    # Clean up original bias field
    original_bias = original_bias.strip()
    
    # Format prompt with the test bias
    prompt = PROMPT_TEMPLATE.format(
        persona_number=f"Persona {row_number + 1:02d}",
        demographics=demographics,
        beliefs=beliefs,
        biases=test_bias  # Use the test bias instead of original
    )
    
    logger.info(f"Processing: User {row_number} ({persona_name}) with bias '{test_bias}'")
    
    # Try with retries
    for retry in range(MAX_RETRIES):
        if retry > 0:
            delay = min(RETRY_DELAY_BASE * (2 ** retry), 30)
            logger.warning(f"Retry {retry}/{MAX_RETRIES} for User {row_number} - {test_bias}")
            await asyncio.sleep(delay)
            
        success, response_text, error = await call_openai_api(
            client, prompt, semaphore, rate_limiter
        )
        
        if success and response_text:
            parsed = parse_json_response(response_text)
            if parsed:
                # Add metadata
                parsed['metadata'] = {
                    'row_number': row_number,
                    'persona_name': persona_name,
                    'original_bias': original_bias,
                    'tested_bias': test_bias,
                    'demographics': demographics,
                    'beliefs': beliefs,
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"✓ Success: User {row_number} with '{test_bias}'")
                return UserBiasProcessingResult(
                    row_number=row_number,
                    persona_name=persona_name,
                    original_bias=original_bias,
                    tested_bias=test_bias,
                    success=True,
                    response=parsed,
                    retry_count=retry
                )
    
    # All retries failed
    logger.error(f"✗ Failed: User {row_number} with '{test_bias}' after {MAX_RETRIES} attempts")
    return UserBiasProcessingResult(
        row_number=row_number,
        persona_name=persona_name,
        original_bias=original_bias,
        tested_bias=test_bias,
        success=False,
        error=f"Failed after {MAX_RETRIES} retries",
        retry_count=MAX_RETRIES
    )

# ======================== RESULT MANAGEMENT ========================

class CrossBiasResultManager:
    """Manages results organized by user ID and bias"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}  # {row_number: {bias: result}}
        self.individual_dir = self.output_dir / "individual_results"
        if INDIVIDUAL_FILES:
            self.individual_dir.mkdir(exist_ok=True)
            
    async def add_result(self, result: UserBiasProcessingResult):
        """Add a result organized by user and bias"""
        if result.row_number not in self.results:
            self.results[result.row_number] = {}
            
        self.results[result.row_number][result.tested_bias] = asdict(result)
        
        # Save individual file if enabled
        if INDIVIDUAL_FILES and result.success:
            await self.save_individual(result)
            
    async def save_individual(self, result: UserBiasProcessingResult):
        """Save individual result file"""
        if not result.success:
            return
            
        # Create user-specific directory
        user_dir = self.individual_dir / f"user_{result.row_number:03d}_{result.persona_name.replace(' ', '_')}"
        user_dir.mkdir(exist_ok=True)
        
        # Save bias-specific file
        filename = user_dir / f"{result.tested_bias.replace(' ', '_').lower()}.json"
        
        output_data = {
            "user_id": result.row_number,
            "persona_name": result.persona_name,
            "original_bias": result.original_bias,
            "tested_bias": result.tested_bias,
            "timestamp": result.timestamp,
            "conversation": result.response.get('conversation', []) if result.response else [],
            "analysis": result.response.get('analysis', {}) if result.response else {},
            "metadata": result.response.get('metadata', {}) if result.response else {}
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(output_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save individual result: {e}")
            
    async def save_consolidated(self):
        """Save all results in a consolidated file"""
        consolidated_file = self.output_dir / "consolidated_cross_bias_results.json"
        
        # Sort by row number
        sorted_results = dict(sorted(self.results.items()))
        
        try:
            with open(consolidated_file, 'w') as f:
                json.dump(sorted_results, f, indent=2)
            logger.debug(f"Saved consolidated results for {len(self.results)} users")
        except Exception as e:
            logger.error(f"Failed to save consolidated results: {e}")

# ======================== MAIN PROCESSING ========================

async def process_cross_bias(resume: bool = True):
    """Main processing function for cross-bias generation"""
    
    # Initialize OpenAI client
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY environment variable not set")
        sys.exit(1)
        
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    # Initialize managers
    checkpoint_mgr = CrossBiasCheckpointManager(CHECKPOINT_FILE)
    result_mgr = CrossBiasResultManager(OUTPUT_DIR)
    
    # Get unique biases
    unique_biases = sorted(ALL_UNIQUE_BIASES)
    
    # Apply test mode limits if enabled
    if TEST_MODE:
        unique_biases = unique_biases[:TEST_BIASES]
        logger.info(f"TEST MODE: Using only {len(unique_biases)} biases")
    else:
        logger.info(f"Found {len(unique_biases)} unique biases")
    
    # Read CSV
    csv_path = Path(CSV_FILE)
    if not csv_path.is_absolute():
        csv_path = Path(__file__).parent / csv_path
        
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        sys.exit(1)
        
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    # Apply test mode limits if enabled
    if TEST_MODE:
        rows = rows[:TEST_ROWS]
        logger.info(f"TEST MODE: Processing only {len(rows)} rows")
        
    total_users = len(rows)
    total_combinations = total_users * len(unique_biases)
    
    logger.info(f"Total users: {total_users}")
    logger.info(f"Total biases: {len(unique_biases)}")
    logger.info(f"Total combinations to process: {total_combinations}")
    
    # Load checkpoint if resuming
    processed_combinations = set()
    checkpoint = None
    
    if resume:
        checkpoint = checkpoint_mgr.load()
        if checkpoint:
            processed_combinations = set(checkpoint.processed_combinations)
            logger.info(f"Resuming: {len(processed_combinations)}/{total_combinations} processed")
    
    # Initialize rate limiting
    semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)
    rate_limiter = RateLimiter(REQUESTS_PER_MINUTE)
    
    # Process combinations
    tasks = []
    task_info = []
    processed_count = len(processed_combinations)
    
    for row_number, row in enumerate(rows):
        for bias in unique_biases:
            # Skip if already processed
            if (row_number, bias) in processed_combinations:
                continue
                
            # Create processing task
            task = process_user_bias_combination(
                client, row, row_number, bias, semaphore, rate_limiter
            )
            tasks.append(task)
            task_info.append((row_number, bias))
            
            # Process in batches
            if len(tasks) >= MAX_CONCURRENT_REQUESTS:
                results = await asyncio.gather(*tasks)
                
                for result, (row_num, bias) in zip(results, task_info):
                    await result_mgr.add_result(result)
                    processed_combinations.add((row_num, bias))
                    processed_count += 1
                    
                # Save checkpoint
                if processed_count % SAVE_INTERVAL == 0:
                    await result_mgr.save_consolidated()
                    
                    checkpoint_data = CrossBiasCheckpoint(
                        processed_combinations=list(processed_combinations),
                        failed_combinations=[(r.row_number, r.tested_bias) 
                                           for r in results if not r.success],
                        total_combinations=total_combinations,
                        total_users=total_users,
                        total_biases=len(unique_biases),
                        start_time=checkpoint.start_time if checkpoint 
                                  else datetime.now().isoformat(),
                        last_update=datetime.now().isoformat()
                    )
                    checkpoint_mgr.save(checkpoint_data)
                    
                # Progress update
                progress = (processed_count / total_combinations) * 100
                logger.info(f"Progress: {processed_count}/{total_combinations} ({progress:.1f}%)")
                
                tasks = []
                task_info = []
    
    # Process remaining tasks
    if tasks:
        results = await asyncio.gather(*tasks)
        for result, (row_num, bias) in zip(results, task_info):
            await result_mgr.add_result(result)
            processed_combinations.add((row_num, bias))
            
    # Final save
    await result_mgr.save_consolidated()
    
    # Generate final output
    final_output = {
        "metadata": {
            "total_users": total_users,
            "total_biases": len(unique_biases),
            "total_combinations": total_combinations,
            "successful_combinations": sum(
                1 for user_results in result_mgr.results.values()
                for result in user_results.values()
                if result['success']
            ),
            "generation_time": datetime.now().isoformat(),
            "model": MODEL_NAME,
            "unique_biases": unique_biases
        },
        "results_by_user": result_mgr.results
    }
    
    # Save final output
    final_path = Path(OUTPUT_DIR) / FINAL_OUTPUT_FILE
    with open(final_path, 'w') as f:
        json.dump(final_output, f, indent=2)
        
    # Summary
    logger.info("=" * 60)
    logger.info("CROSS-BIAS PROCESSING COMPLETE")
    logger.info(f"Total combinations processed: {len(processed_combinations)}")
    logger.info(f"Final output: {final_path}")
    if INDIVIDUAL_FILES:
        logger.info(f"Individual files: {result_mgr.individual_dir}")
    
    # Clean up checkpoint
    checkpoint_mgr.delete()

# ======================== SIGNAL HANDLING ========================

def signal_handler(signum, frame):
    """Handle interruption gracefully"""
    logger.info("\nInterruption received. Checkpoint saved. Run script again to resume.")
    sys.exit(0)

# ======================== ENTRY POINT ========================

async def main():
    """Main entry point"""
    # Set up signal handling
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("=" * 60)
    logger.info("OPENAI CROSS-BIAS PROCESSOR")
    logger.info("Generating conversations for each user with every unique bias")
    logger.info("=" * 60)
    
    # Test connection
    if not OPENAI_API_KEY:
        logger.error("Please set OPENAI_API_KEY environment variable")
        sys.exit(1)
        
    # Check for resume
    checkpoint_mgr = CrossBiasCheckpointManager(CHECKPOINT_FILE)
    checkpoint = checkpoint_mgr.load()
    
    if checkpoint:
        logger.info(f"Found checkpoint from {checkpoint.last_update}")
        logger.info(f"Processed: {len(checkpoint.processed_combinations)}/{checkpoint.total_combinations}")
        response = input("Resume from checkpoint? (y/n): ").strip().lower()
        resume = response == 'y'
    else:
        resume = False
        
    # Start processing
    await process_cross_bias(resume=resume)

if __name__ == "__main__":
    asyncio.run(main())