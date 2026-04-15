#!/usr/bin/env zsh

set -euo pipefail

# Debug wrapper: keep generative evaluation active but skip the two
# native-heavy metrics most likely to trigger runtime crashes.

script_dir=${0:A:h}

print -r -- "DEBUG variant: skip both sbm and orbit metrics."

exec "${script_dir}/run-upstream-digress-sbm-modal.zsh" \
  "force_fresh=true" \
  "model.evaluator.skip_metrics=[sbm,orbit]" \
  "$@"
