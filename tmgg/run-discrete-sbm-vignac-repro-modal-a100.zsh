#!/usr/bin/env zsh
# Vignac / GDPO-style DiGress SBM pretraining repro on Modal A100-40GB.
#
# Thin wrapper over ``tmgg-modal run tmgg-discrete-gen``. All parity knobs
# (lr, weight decay, amsgrad, max_steps, val cadence, K=1 dims,
# extra_features='all', dim_ffy=2048, seed=666) live in
# ``src/tmgg/experiments/exp_configs/experiment/discrete_sbm_vignac_repro.yaml``,
# so this script only sets execution-time knobs (GPU tier, precision,
# detach, wandb credentials).
#
# Defaults: A100-40GB, bf16-mixed (~6-8 h vs ~12-15 h fp32; A100's native
# bf16 tensor cores keep fp32's exponent range so no loss-scaling needed).
# Override ``PRECISION=32`` for byte-faithful fp32.
#
# See ``docs/debugging-modal.md`` for live-log streaming guidance.

set -euo pipefail

: "${USE_DOPPLER:=1}"
: "${DEPLOY_FIRST:=1}"
: "${DETACH:=1}"
: "${DRY_RUN:=0}"
: "${GPU_TIER:=fast}"            # A100-40GB
: "${PRECISION:=bf16-mixed}"     # set to 32 for byte-faithful fp32
# MODAL_DEBUG=0 (default) → container sets PYTHONOPTIMIZE=1 inside the
# training subprocess, stripping ``assert`` and ``if __debug__:`` blocks
# from the hot path (~50 host-side syncs/step removed; see
# docs/reports/2026-04-28-sync-review/99-synthesis.md).
# MODAL_DEBUG=1 → asserts active, for numerical investigation only.
: "${MODAL_DEBUG:=0}"
: "${MPLCONFIGDIR:=${TMPDIR:-/tmp}/tmgg-mpl-cache}"

mkdir -p "${MPLCONFIGDIR}"
export MPLCONFIGDIR

run_prefixed() {
  if [[ "${USE_DOPPLER}" == "1" ]]; then
    doppler run -- "$@"
  else
    "$@"
  fi
}

if [[ "${DEPLOY_FIRST}" == "1" ]]; then
  print -r -- "Deploying Modal app and refreshing secrets..."
  run_prefixed mise run modal-deploy
fi

if [[ "${MODAL_DEBUG}" == "1" ]]; then
  modal_debug_override=true
else
  modal_debug_override=false
fi

typeset -a cmd
cmd=(
  uv run tmgg-modal run tmgg-discrete-gen
  +experiment=discrete_sbm_vignac_repro
  trainer.precision="${PRECISION}"
  modal_debug="${modal_debug_override}"
  --gpu "${GPU_TIER}"
)

if [[ "${DETACH}" == "1" ]]; then
  cmd+=(--detach)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  cmd+=(--dry-run)
fi

if (( $# > 0 )); then
  cmd+=("$@")
fi

print -r -- "Launching Vignac SBM repro on Modal (${GPU_TIER}, ${PRECISION})..."
printf ' %q' "${cmd[@]}"
print

run_prefixed "${cmd[@]}"
