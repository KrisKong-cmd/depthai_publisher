#!/usr/bin/env python3

"""
Wrapper script for dai_publisher_yolov5_runner.py that handles device busy errors
"""

import subprocess
import time
import sys
import os
import signal

def kill_existing_processes():
    """Kill any existing DepthAI processes"""
    print("Checking for existing DepthAI processes...")
    try:
        # Kill any existing runner processes
        subprocess.run(["pkill", "-f", "dai_publisher_yolov5_runner"], stderr=subprocess.DEVNULL)
        time.sleep(1)
    except:
        pass

def reset_usb_device():
    """Try to reset the USB device"""
    print("Attempting to reset DepthAI USB device...")
    try:
        # Find the device
        result = subprocess.run(["lsusb"], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if "Movidius" in line or "Luxonis" in line or "03e7" in line:
                print(f"Found device: {line}")
                # Extract bus and device numbers
                parts = line.split()
                if len(parts) >= 4:
                    bus = parts[1]
                    device = parts[3].rstrip(':')
                    device_path = f"/dev/bus/usb/{bus}/{device}"
                    # Try to reset (might need sudo)
                    subprocess.run(["sudo", "usbreset", device_path], stderr=subprocess.DEVNULL)
                    print("USB reset attempted")
                    time.sleep(2)
                    return True
    except Exception as e:
        print(f"Could not reset USB: {e}")
    return False

def run_depthai_node(max_retries=3):
    """Run the DepthAI node with retry logic"""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    runner_script = os.path.join(script_dir, "dai_publisher_yolov5_runner.py")
    
    if not os.path.exists(runner_script):
        print(f"Error: Cannot find {runner_script}")
        sys.exit(1)
    
    for attempt in range(max_retries):
        print(f"\nAttempt {attempt + 1} of {max_retries}...")
        
        if attempt > 0:
            # Clean up before retry
            kill_existing_processes()
            reset_usb_device()
            time.sleep(2)
        
        try:
            # Run the actual node
            print(f"Starting DepthAI node...")
            proc = subprocess.Popen(
                ["python3", runner_script],
                stdout=sys.stdout,
                stderr=sys.stderr
            )
            
            # Wait for the process
            proc.wait()
            
            # If we get here, process ended
            if proc.returncode == 0:
                print("DepthAI node exited normally")
                break
            else:
                print(f"DepthAI node exited with code {proc.returncode}")
                
                # Check if it's the device busy error
                if attempt < max_retries - 1:
                    print("Will retry in 3 seconds...")
                    time.sleep(3)
        
        except KeyboardInterrupt:
            print("\nShutting down...")
            if proc:
                proc.terminate()
            sys.exit(0)
        
        except Exception as e:
            print(f"Error running node: {e}")
            if attempt < max_retries - 1:
                print("Will retry...")
            else:
                print("Max retries reached. Exiting.")
                sys.exit(1)

def signal_handler(sig, frame):
    print('\nShutdown requested...')
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("=== DepthAI Node Wrapper ===")
    kill_existing_processes()
    run_depthai_node()