#!/bin/bash

# Check if pigpio daemon is already running
if pgrep -x "pigpiod" > /dev/null
then
    echo "pigpio daemon is already running"
else
    echo "Starting pigpio daemon..."
    # Try to start without sudo first (if user has permissions)
    pigpiod 2>/dev/null
    
    # If that failed, try with sudo
    if [ $? -ne 0 ]; then
        echo "Trying with sudo..."
        sudo pigpiod 2>/dev/null
    fi
    
    sleep 1
    
    # Verify it started
    if pgrep -x "pigpiod" > /dev/null
    then
        echo "pigpio daemon started successfully"
    else
        echo "WARNING: Failed to start pigpio daemon - it may need to be started manually with: sudo pigpiod"
        echo "Continuing anyway as it might already be running..."
    fi
fi

# Keep the node alive but don't block
echo "pigpio daemon check complete"
# Exit successfully so launch file continues
exit 0