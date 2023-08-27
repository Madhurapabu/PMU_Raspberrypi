import subprocess
import time
import signal
import sys
import random
import string

def stop_script(signal, frame):
    # Clean up and terminate tmux session
    subprocess.run(["tmux", "kill-session", "-t", session_name])
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, stop_script)

# Generate a random session name
session_name = ''.join(random.choices(string.ascii_lowercase, k=8))

# Define the commands to run the Python scripts
script1_command = "python 3ph_phasor_estimation.py"
script2_command = "python 2.py"

# Create a new tmux session
subprocess.run(["tmux", "new-session", "-d", "-s", session_name])

# Split the terminal horizontally
subprocess.run(["tmux", "split-window", "-h", "-t", f"{session_name}:0"])

# Run the first script in the left pane
subprocess.run(["tmux", "send-keys", "-t", f"{session_name}:0.0", script1_command, "C-m"])

# Run the second script in the right pane
subprocess.run(["tmux", "send-keys", "-t", f"{session_name}:0.1", script2_command, "C-m"])

# Attach to the tmux session (optional)
subprocess.run(["tmux", "attach-session", "-t", session_name])

# Wait for some time to allow scripts to run (adjust as needed)
time.sleep(0.02)
