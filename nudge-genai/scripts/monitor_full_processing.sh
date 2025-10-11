#!/bin/bash
# Monitor full cross-bias processing progress

echo "Full Cross-Bias Processing Monitor"
echo "=================================="
echo "Target: 100 users × 25 biases = 2,500 conversations"
echo "Started: $(date)"
echo ""

LOG_FILE="cross_bias_full_processing.log"
TOTAL_TARGET=2500

while true; do
    # Count successes and failures
    SUCCESS_COUNT=$(grep -c "✓ Success:" "$LOG_FILE" 2>/dev/null || echo "0")
    FAILURE_COUNT=$(grep -c "✗ Failed" "$LOG_FILE" 2>/dev/null || echo "0")
    
    # Calculate totals and percentage
    TOTAL_PROCESSED=$((SUCCESS_COUNT + FAILURE_COUNT))
    PERCENTAGE=$((TOTAL_PROCESSED * 100 / TOTAL_TARGET))
    
    # Estimate time remaining (assuming ~2 seconds per request average)
    REMAINING=$((TOTAL_TARGET - TOTAL_PROCESSED))
    EST_SECONDS=$((REMAINING * 2))
    EST_MINUTES=$((EST_SECONDS / 60))
    
    # Get current user being processed
    CURRENT_USER=$(grep "Processing: User" "$LOG_FILE" | tail -1 | sed 's/.*User \([0-9]*\).*/\1/' || echo "0")
    CURRENT_BIAS=$(grep "Processing:" "$LOG_FILE" | tail -1 | sed "s/.*with bias '\([^']*\)'.*/\1/" || echo "Starting...")
    
    # Display status
    printf "\r[%s] Progress: %d/%d (%d%%) | User: %d/99 | Bias: %s | Success: %d | Failed: %d | ETA: ~%d min" \
        "$(date +%H:%M:%S)" \
        "$TOTAL_PROCESSED" \
        "$TOTAL_TARGET" \
        "$PERCENTAGE" \
        "$CURRENT_USER" \
        "${CURRENT_BIAS:0:20}" \
        "$SUCCESS_COUNT" \
        "$FAILURE_COUNT" \
        "$EST_MINUTES"
    
    # Check if complete
    if [ "$TOTAL_PROCESSED" -ge "$TOTAL_TARGET" ]; then
        echo ""
        echo "Processing complete!"
        echo "Final stats: Success=$SUCCESS_COUNT, Failed=$FAILURE_COUNT"
        break
    fi
    
    # Check if process is still running
    if ! ps aux | grep -q "[o]penai_cross_bias_processor.py"; then
        echo ""
        echo "Process has stopped. Total processed: $TOTAL_PROCESSED"
        break
    fi
    
    sleep 10
done

echo ""
echo "Check results in: synthetic_data_cross_bias_output/"