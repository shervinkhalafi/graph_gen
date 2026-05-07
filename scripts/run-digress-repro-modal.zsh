#!/usr/bin/env zsh
# DiGress repro panel launcher.
#
# Usage:
#   ./scripts/run-digress-repro-modal.zsh <dataset-key> [hydra-overrides...]
#
# Dataset keys for the paper Table 2 panel:
#   sbm                          | DiGress baseline on SPECTRE-SBM
#   sbm-pearl                    | + R-PEARL features
#   sbm-pearl-spectral           | + R-PEARL + spectral attention
#   sbm-pearl-gnnconv-norm       | + R-PEARL + GCAT (D^-1/2 A D^-1/2)
#   enzymes                      | DiGress baseline on PyG ENZYMES
#   enzymes-pearl                | + R-PEARL
#   enzymes-pearl-spectral       | + R-PEARL + spectral attention
#   enzymes-pearl-gnnconv-norm   | + R-PEARL + GCAT
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
  sbm)                         EXP="discrete_sbm_vignac_repro_exact" ;;
  sbm-pearl)                   EXP="discrete_sbm_pearl_repro_exact" ;;
  sbm-pearl-spectral)          EXP="discrete_sbm_pearl_spectral_repro_exact" ;;
  sbm-pearl-gnnconv-norm)      EXP="discrete_sbm_pearl_gnnconv_norm_repro_exact" ;;
  enzymes)                     EXP="discrete_enzymes_vignac_repro_exact" ;;
  enzymes-pearl)               EXP="discrete_enzymes_pearl_repro_exact" ;;
  enzymes-pearl-spectral)      EXP="discrete_enzymes_pearl_spectral_repro_exact" ;;
  enzymes-pearl-gnnconv-norm)  EXP="discrete_enzymes_pearl_gnnconv_norm_repro_exact" ;;
  *) echo "unknown dataset: $DATASET" >&2; exit 1 ;;
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
