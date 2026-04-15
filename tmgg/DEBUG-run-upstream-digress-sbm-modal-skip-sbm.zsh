#!/usr/bin/env zsh

set -euo pipefail

# Debug wrapper: isolate the graph-tool / SBM evaluation path by
# skipping only sbm_accuracy.

script_dir=${0:A:h}

print -r -- "DEBUG variant: skip sbm metric only."

exec "${script_dir}/run-upstream-digress-sbm-modal.zsh" \
  "force_fresh=true" \
  "model.evaluator.skip_metrics=[sbm]" \
  "$@"
