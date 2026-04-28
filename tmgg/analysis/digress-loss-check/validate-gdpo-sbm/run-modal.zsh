#!/usr/bin/env zsh
# Spawn a GDPO-SBM sanity-check call on the deployed Modal app.
#
# Prereq: the app must be deployed (re-run after changes to validate.py
# or the checkpoint, both baked at image-build time):
#   ./analysis/digress-loss-check/validate-gdpo-sbm/deploy-modal.zsh
#
# Usage:
#   ./run-modal.zsh                                # defaults: A10G, 40 samples
#   ./run-modal.zsh --gpu a100 --num-samples 200   # bigger run
#   ./run-modal.zsh --gpu t4                       # cheapest, slower
#
# Prints a FunctionCall id; the terminal can be closed immediately, the
# call keeps running on Modal. Fetch the result later with:
#   uv run python analysis/digress-loss-check/validate-gdpo-sbm/client.py fetch \
#     --call-id <id> [--out-dir <path>] [--timeout 0]
# or pull the outputs from the ``tmgg-outputs`` volume:
#   modal volume get tmgg-outputs /data/outputs/validate-gdpo-sbm/

set -euo pipefail

script_dir=${0:A:h}
repo_root=${script_dir:A:h:h:h}  # validate-gdpo-sbm -> digress-loss-check -> analysis -> tmgg

cd "$repo_root"

exec uv run --with modal --with click python \
  analysis/digress-loss-check/validate-gdpo-sbm/client.py spawn "$@"
