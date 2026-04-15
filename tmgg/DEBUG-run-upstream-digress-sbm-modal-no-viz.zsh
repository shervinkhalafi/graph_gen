#!/usr/bin/env zsh

set -euo pipefail

# Debug wrapper: disable validation visualizations while keeping
# generative evaluation metrics enabled.

script_dir=${0:A:h}

print -r -- "DEBUG variant: disable validation visualizations only."

exec "${script_dir}/run-upstream-digress-sbm-modal.zsh" \
  "force_fresh=true" \
  "evaluation.visualization.enabled=false" \
  "$@"
