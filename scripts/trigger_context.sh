#!/bin/bash
# Trigger on-demand context generation

# Method 1: File trigger (works regardless of how Contextor was started)
touch /tmp/contextor_trigger
echo "Context generation triggered via file"

# Method 2: Signal (if running as a process)
# Uncomment if you prefer signal-based triggering
# pkill -USR1 -f "python.*contextor"
# echo "Context generation triggered via signal"
