#!/bin/bash
# Monitor cross-bias processing progress

echo "Cross-Bias Processing Monitor"
echo "============================"
echo "Target: 10 users × 25 biases = 250 conversations"
echo ""

while true; do
    # Count successes
    SUCCESS_COUNT=$(grep -c "✓ Success:" cross_bias_processing.log 2>/dev/null || echo "0")
    
    # Count failures
    FAILURE_COUNT=$(grep -c "✗ Failed" cross_bias_processing.log 2>/dev/null || echo "0")
    
    # Calculate percentage
    TOTAL_PROCESSED=$((SUCCESS_COUNT + FAILURE_COUNT))
    PERCENTAGE=$((TOTAL_PROCESSED * 100 / 250))
    
    # Get current processing info
    CURRENT=$(tail -n 1 cross_bias_processing.log | grep "Processing:" || echo "Waiting...")
    
    # Clear line and print status
    printf "\r[%s] Progress: %d/250 (%d%%) | Success: %d | Failed: %d | Current: %s" \
        "$(date +%H:%M:%S)" \
        "$TOTAL_PROCESSED" \
        "$PERCENTAGE" \
        "$SUCCESS_COUNT" \
        "$FAILURE_COUNT" \
        "${CURRENT:0:60}..."
    
    # Check if complete
    if [ "$TOTAL_PROCESSED" -ge 250 ]; then
        echo ""
        echo "Processing complete!"
        break
    fi
    
    # Check if process is still running
    if ! ps aux | grep -q "[o]penai_cross_bias_processor.py"; then
        echo ""
        echo "Process has stopped. Check log for details."
        break
    fi
    
    sleep 5
done