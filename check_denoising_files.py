#!/usr/bin/env python3
"""Check basedpyright output for only the 4 denoising files."""

import subprocess
import sys

files = [
    "denoising/main.py",
    "denoising/src/models/gnn.py",
    "denoising/src/models/attention.py",
    "denoising/src/data/sbm.py",
]

# Run basedpyright
result = subprocess.run(
    ["uv", "run", "basedpyright"] + files,
    capture_output=True,
    text=True,
)

# Filter output to only show lines related to our target files
lines = result.stdout.split("\n") + result.stderr.split("\n")
denoising_lines = []
capture = False

for line in lines:
    # Check if line starts with one of our files
    if any(line.startswith(f) for f in files):
        capture = True
    elif line.startswith("/") and not any(f in line for f in files):
        # Stop capturing when we hit a different file
        capture = False

    if capture and line.strip():
        denoising_lines.append(line)

# Print results
if denoising_lines:
    print("Issues in denoising files:")
    print("\n".join(denoising_lines))
else:
    print("No issues found in denoising files!")

# Get summary
for line in lines:
    if "errors" in line and "warnings" in line:
        print(f"\nOverall: {line}")

sys.exit(result.returncode)
