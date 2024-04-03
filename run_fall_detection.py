import os
import subprocess

# Change directory to Human-Falling-Detect-Tracks
os.chdir("Human-Falling-Detect-Tracks")

# Command to run the main.py script with the specified arguments
command = "python main.py -C 0 --show_detected --device cpu"

# Run the command using subprocess
subprocess.run(command, shell=True)