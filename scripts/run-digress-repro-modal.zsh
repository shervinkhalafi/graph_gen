#!/usr/bin/env zsh
# DiGress repro panel launcher.
#
# Usage:
#   ./scripts/run-digress-repro-modal.zsh <cell-key> [hydra-overrides...]
#
# Cell keys for the paper Table 2 panel (matches the names in
# paper-artifacts/repro-ablations/configs/):
#   digress-sbm                       | DiGress baseline on SPECTRE-SBM
#   digress-pearl-sbm                 | + R-PEARL features
#   digress-pearl-spectral-sbm        | + R-PEARL + spectral attention
#   digress-pearl-gcat-sbm            | + R-PEARL + GCAT (D^-1/2 A D^-1/2)
#   digress-enzymes                   | DiGress baseline on PyG ENZYMES
#   digress-pearl-enzymes             | + R-PEARL
#   digress-pearl-spectral-enzymes    | + R-PEARL + spectral attention
#   digress-pearl-gcat-enzymes        | + R-PEARL + GCAT
#
# Prereq: deploy the Modal app once before the first launch:
#   uv run modal deploy -m tmgg.modal._functions
#
# Env knobs (with defaults):
#   DEPLOY_FIRST=0 — set to 1 to (re)deploy Modal apps before launch
#   DETACH=1, DRY_RUN=0, GPU_TIER=fast, PRECISION=bf16-mixed, MODAL_DEBUG=0
#
# Requires WANDB_API_KEY in the environment for run logging.

set -euo pipefail

: "${DEPLOY_FIRST:=0}"
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
  digress-sbm)                  EXP="digress_sbm" ;;
  digress-pearl-sbm)            EXP="digress_pearl_sbm" ;;
  digress-pearl-spectral-sbm)   EXP="digress_pearl_spectral_sbm" ;;
  digress-pearl-gcat-sbm)       EXP="digress_pearl_gcat_sbm" ;;
  digress-enzymes)              EXP="digress_enzymes" ;;
  digress-pearl-enzymes)        EXP="digress_pearl_enzymes" ;;
  digress-pearl-spectral-enzymes) EXP="digress_pearl_spectral_enzymes" ;;
  digress-pearl-gcat-enzymes)   EXP="digress_pearl_gcat_enzymes" ;;
  *) echo "unknown cell: $DATASET" >&2; exit 1 ;;
esac

run_prefixed() {
  "$@"
}

if [[ "${DEPLOY_FIRST}" == "1" ]]; then
  print -r -- "Deploying Modal apps..."
  for module in tmgg.modal._functions tmgg.modal._eval_all_functions; do
    run_prefixed uv run modal deploy -m "${module}"
  done
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
