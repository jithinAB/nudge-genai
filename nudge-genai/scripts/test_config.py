#!/usr/bin/env python3
"""
Configuration test script to verify settings before running the main processor
"""

import sys
from pathlib import Path

def test_configuration():
    """Test and display current configuration"""

    print("=" * 60)
    print("LM STUDIO PROCESSOR - CONFIGURATION CHECK")
    print("=" * 60)

    # Import configuration from main script
    sys.path.insert(0, str(Path(__file__).parent))
    from lm_studio_processor import (
        MAX_CONCURRENT_REQUESTS,
        REQUESTS_PER_MINUTE,
        MAX_RETRIES,
        RETRY_DELAY_BASE,
        REQUEST_TIMEOUT,
        SAVE_INTERVAL,
        TEMPERATURE,
        MAX_TOKENS,
        CSV_FILE,
        OUTPUT_DIR,
        LM_STUDIO_URL,
        MODEL_NAME
    )

    print("\n📋 PROCESSING CONFIGURATION:")
    print(f"  • Max Concurrent Requests: {MAX_CONCURRENT_REQUESTS} (sequential processing)")
    print(f"  • Requests Per Minute: {REQUESTS_PER_MINUTE}")
    print(f"  • Request interval: {60/REQUESTS_PER_MINUTE:.1f} seconds between requests")
    print(f"  • Max Retries: {MAX_RETRIES}")
    print(f"  • Retry Delay Base: {RETRY_DELAY_BASE} seconds")
    print(f"  • Request Timeout: {REQUEST_TIMEOUT} seconds ({REQUEST_TIMEOUT/60:.1f} minutes)")
    print(f"  • Save Interval: Every {SAVE_INTERVAL} row(s)")

    print("\n🤖 MODEL CONFIGURATION:")
    print(f"  • LM Studio URL: {LM_STUDIO_URL}")
    print(f"  • Model Name: {MODEL_NAME}")
    print(f"  • Temperature: {TEMPERATURE}")
    print(f"  • Max Tokens: {MAX_TOKENS}")

    print("\n📁 FILE PATHS:")
    print(f"  • CSV File: {CSV_FILE}")
    print(f"  • Output Directory: {OUTPUT_DIR}")

    # Check CSV file exists
    csv_path = Path(__file__).parent / CSV_FILE
    if csv_path.exists():
        print(f"  ✓ CSV file found at: {csv_path}")
        # Count rows
        import csv
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            row_count = len(list(reader))
        print(f"  • Total rows in CSV: {row_count}")
        print(f"  • Will process first 10 rows (test mode)")
    else:
        print(f"  ✗ CSV file not found at: {csv_path}")

    print("\n⏱️ ESTIMATED TIMING:")
    print(f"  • Min time per row: {60/REQUESTS_PER_MINUTE:.1f} seconds (rate limit)")
    print(f"  • Max time per row: {REQUEST_TIMEOUT} seconds (timeout)")
    print(f"  • Estimated time for 10 rows: {10 * (60/REQUESTS_PER_MINUTE):.1f} - {10 * 30:.0f} seconds")
    print(f"  • With retries: up to {10 * MAX_RETRIES * 30:.0f} seconds worst case")

    print("\n✅ OPTIMIZATIONS FOR STABILITY:")
    print("  1. Sequential processing (1 request at a time)")
    print("  2. Extended timeout for complex reasoning")
    print("  3. Progressive retry backoff")
    print("  4. Checkpoint after every row")
    print("  5. Rate limited to prevent overload")

    print("\n" + "=" * 60)
    print("Ready to process! Run: python lm_studio_processor.py")
    print("=" * 60)

if __name__ == "__main__":
    test_configuration()