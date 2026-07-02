#!/bin/bash

# Configurable threshold (percentage of GPU memory)
THRESHOLD_PERCENT=30

echo "Checking for processes using more than $THRESHOLD_PERCENT% of GPU memory..."

# Get list of all GPU indices
gpu_indices=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)

for gpu_index in $gpu_indices; do
    echo "Checking GPU $gpu_index..."
    
    # Get total memory for this GPU
    total_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $gpu_index)
    
    # Calculate threshold in MiB
    threshold=$(echo "$total_memory * $THRESHOLD_PERCENT / 100" | bc)
    
    # Get processes on this GPU using more than threshold memory
    nvidia-smi --query-compute-apps=pid,used_memory,gpu_bus_id --format=csv,noheader,nounits -i $gpu_index | \
    while IFS=',' read -r pid used_memory bus_id; do
        # Clean up any whitespace
        pid=$(echo $pid | xargs)
        used_memory=$(echo $used_memory | xargs)
        
        if [ "$used_memory" -gt "$threshold" ]; then
            # Get process name for better logging
            process_name=$(ps -p $pid -o comm= 2>/dev/null || echo "unknown process")
            echo "GPU $gpu_index: Killing PID $pid ($process_name) using ${used_memory}MiB (> ${threshold}MiB threshold)"
            kill -9 "$pid" 2>/dev/null
            
            # # Verify if process was killed
            # if ps -p $pid > /dev/null; then
            #     echo "Warning: Failed to kill PID $pid"
            # fi
        fi
    done
done

echo "Zombie process cleanup complete."