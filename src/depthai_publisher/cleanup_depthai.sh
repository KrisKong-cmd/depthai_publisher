#!/bin/bash

echo "=== DepthAI Cleanup Script ==="

# Kill any existing DepthAI processes
echo "Stopping any existing DepthAI processes..."
pkill -f dai_publisher_yolov5_runner.py 2>/dev/null
pkill -f test_depthai.py 2>/dev/null
pkill -f aruco_detection 2>/dev/null

# Give processes time to terminate
sleep 2

# Check if any are still running
if pgrep -f "dai_publisher_yolov5" > /dev/null; then
    echo "Force killing stubborn processes..."
    pkill -9 -f dai_publisher_yolov5_runner.py 2>/dev/null
    sleep 1
fi

# Try to reset the USB device
echo "Looking for DepthAI device..."
DEVICE=$(lsusb | grep -E "Movidius|Luxonis|03e7" | head -n1)

if [ ! -z "$DEVICE" ]; then
    echo "Found device: $DEVICE"
    
    # Extract bus and device numbers
    BUS=$(echo $DEVICE | sed 's/Bus \([0-9]*\).*/\1/')
    DEV=$(echo $DEVICE | sed 's/.*Device \([0-9]*\).*/\1/')
    
    if [ ! -z "$BUS" ] && [ ! -z "$DEV" ]; then
        echo "Resetting USB device on Bus $BUS Device $DEV..."
        
        # Try usbreset if available
        if command -v usbreset &> /dev/null; then
            sudo usbreset "/dev/bus/usb/$BUS/$DEV" 2>/dev/null
        else
            # Alternative reset method
            echo "Attempting USB power cycle..."
            echo "$BUS-$DEV" | sudo tee /sys/bus/usb/drivers/usb/unbind > /dev/null 2>&1
            sleep 1
            echo "$BUS-$DEV" | sudo tee /sys/bus/usb/drivers/usb/bind > /dev/null 2>&1
        fi
        
        echo "Waiting for device to reinitialize..."
        sleep 3
    fi
else
    echo "No DepthAI device found via lsusb"
fi

echo "Cleanup complete!"
echo ""
echo "You can now start the DepthAI nodes."