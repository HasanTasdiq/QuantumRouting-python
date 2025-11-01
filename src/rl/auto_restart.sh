#!/bin/bash

echo "Starting auto-restart script for dist_agent.py (every 60 minutes)..."
# Find and kill any process using port 8000
PORT_PID=$(lsof -t -i:8000)
MEMORY_LIMIT_GB=50  # Set memory limit to 4GB

if [ ! -z "$PORT_PID" ]; then
    echo "Killing process on port 8000 (PID: $PORT_PID)"
    kill -9 $PORT_PID
    sleep 1
fi

while true; do
    echo "=== $(date) ==="
    
    # PORT_PID=$(lsof -t -i:8000)
    # if [ ! -z "$PORT_PID" ]; then
    #     echo "Killing process on port 8000 (PID: $PORT_PID)"
    #     kill -9 $PORT_PID
    #     sleep 1
    # fi

    # Kill specific Python processes (dist_agent.py)
    PYTHON_PID=$(pgrep -f "dist_agent.py")
    if [ ! -z "$PYTHON_PID" ]; then
        echo "Killing existing dist_agent.py process (PID: $PYTHON_PID)"
        
        # Try graceful shutdown first
        kill -TERM $PYTHON_PID
        sleep 2
        
        # Force kill if still running
        if kill -0 $PYTHON_PID 2>/dev/null; then
            echo "Process still running, forcing kill..."
            kill  $PYTHON_PID
        fi
        
        # Wait for process to fully terminate
        echo "Waiting for process to terminate..."
        while kill -0 $PYTHON_PID 2>/dev/null; do
            sleep 0.5
        done
        echo "Process terminated successfully"
        
        # Wait for system to reclaim memory
        echo "Waiting for memory to be reclaimed..."
        sleep 5
    fi

    echo "Starting dist_agent.py..."
    # sleep 5
    python3 dist_agent.py &
    
    # echo "Waiting 60 minutes before next restart..."
    # sleep 20  # 3600 seconds = 60 minutes

    while true; do
        sleep 10
        PYTHON_PID=$(pgrep -f "dist_agent.py")

        MEM_KB=$(ps -o rss= -p $PYTHON_PID 2>/dev/null)
        if [ -n "$MEM_KB" ]; then
            MEM_MB=$((MEM_KB / 1024))
            MEM_GB=$((MEM_MB / 1024))
            echo "$(date): Current memory usage: (${MEM_GB}"
            if [ $MEM_MB -gt $MEMORY_LIMIT_GB ]; then
                echo "$(date): Memory usage ${MEM_GB}GB > ${MEMORY_LIMIT_GB}GB. Restarting..."
                break
            fi
        fi
    done
done