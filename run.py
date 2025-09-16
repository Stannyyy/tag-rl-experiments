import subprocess

# Run the experiment part-by-part to prevent intermittant memory issues
while True:
    result = subprocess.run(["python", "main.py", "experiment-20250915-1609"])
    if result.returncode != 0:
        break