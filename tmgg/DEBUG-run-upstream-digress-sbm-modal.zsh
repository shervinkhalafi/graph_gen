#!/usr/bin/env zsh
# Debug wrapper around run-upstream-digress-sbm-modal.zsh.
#
# Drives the upstream parity launcher with one of four isolation
# strategies for the native-heavy generative-evaluation metrics most
# likely to trigger runtime crashes under Modal. Pick a mode, append
# any extra Hydra overrides.
#
# Usage:
#   DEBUG-run-upstream-digress-sbm-modal.zsh <mode> [hydra overrides...]
#
# Modes:
#   no-viz          disable validation visualizations; keep all metrics
#   skip-orbit      skip the ORCA / orbit-MMD metric only
#   skip-sbm        skip the graph-tool SBM-accuracy metric only
#   skip-sbm-orbit  skip both graph-tool SBM and ORCA orbit metrics

set -euo pipefail

if (( $# < 1 )); then
  print -u2 -r -- "usage: $0 <no-viz|skip-orbit|skip-sbm|skip-sbm-orbit> [hydra overrides...]"
  exit 2
fi

mode=$1
shift

case $mode in
  no-viz)         override="evaluation.visualization.enabled=false" ;;
  skip-orbit)     override="model.evaluator.skip_metrics=[orbit]" ;;
  skip-sbm)       override="model.evaluator.skip_metrics=[sbm]" ;;
  skip-sbm-orbit) override="model.evaluator.skip_metrics=[sbm,orbit]" ;;
  *)
    print -u2 -r -- "unknown mode: $mode (expected no-viz|skip-orbit|skip-sbm|skip-sbm-orbit)"
    exit 2
    ;;
esac

script_dir=${0:A:h}
print -r -- "DEBUG variant: $mode ($override)"

exec "${script_dir}/run-upstream-digress-sbm-modal.zsh" \
  "force_fresh=true" \
  "$override" \
  "$@"
