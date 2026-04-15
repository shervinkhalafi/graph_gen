#!/usr/bin/env zsh

set -euo pipefail

# Debug wrapper: isolate the ORCA / orbit-MMD path by
# skipping only orbit_mmd.

script_dir=${0:A:h}

print -r -- "DEBUG variant: skip orbit metric only."

exec "${script_dir}/run-upstream-digress-sbm-modal.zsh" \
  "force_fresh=true" \
  "model.evaluator.skip_metrics=[orbit]" \
  "$@"
