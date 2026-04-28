#!/usr/bin/env zsh
# DiGress repro panel launcher.
#
# Usage:
#   ./scripts/run-digress-repro-modal.zsh <sbm|planar|qm9|moses|guacamol> [hydra-overrides...]
#
# Env knobs (with defaults):
#   USE_DOPPLER=1, DEPLOY_FIRST=1, DETACH=1, DRY_RUN=0,
#   GPU_TIER=fast, PRECISION=bf16-mixed, MODAL_DEBUG=0
#
# See docs/specs/2026-04-28-digress-repro-datasets-spec.md for context.

set -euo pipefail

: "${USE_DOPPLER:=1}"
: "${DEPLOY_FIRST:=1}"
: "${DETACH:=1}"
: "${DRY_RUN:=0}"
: "${GPU_TIER:=fast}"
: "${PRECISION:=bf16-mixed}"
: "${MODAL_DEBUG:=0}"
: "${MPLCONFIGDIR:=${TMPDIR:-/tmp}/tmgg-mpl-cache}"

mkdir -p "${MPLCONFIGDIR}"
export MPLCONFIGDIR

DATASET="${1:?usage: $0 <sbm|planar|qm9|moses|guacamol> [hydra-overrides...]}"
shift

case "$DATASET" in
  sbm)        EXP="discrete_sbm_vignac_repro" ;;
  planar)     EXP="discrete_planar_digress_repro" ;;
  qm9)        EXP="discrete_qm9_digress_repro" ;;
  moses)      EXP="discrete_moses_digress_repro" ;;
  guacamol)   EXP="discrete_guacamol_digress_repro" ;;
  *) echo "unknown dataset: $DATASET" >&2; exit 1 ;;
esac

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
  +experiment="${EXP}"
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

print -r -- "Launching DiGress repro: dataset=${DATASET} (${GPU_TIER}, ${PRECISION})"
printf ' %q' "${cmd[@]}"
print

run_prefixed "${cmd[@]}"
