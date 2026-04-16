#!/usr/bin/env zsh

set -euo pipefail

# Launch the upstream-DiGress SBM training run on Modal.
#
# Uses the SPECTRE 200-graph SBM fixture (variable n in [44, 187],
# 2-5 communities, p_intra=0.3, p_inter=0.005) — the same dataset
# DiGress reports against in Vignac et al. (ICLR 2023). All numerical
# parity fixes from ``docs/reports/2026-04-15-upstream-digress-parity
# -audit.md`` are baked into the configs reached by this command.

: "${USE_DOPPLER:=1}"
: "${DEPLOY_FIRST:=1}"
: "${DETACH:=1}"
: "${DRY_RUN:=0}"
: "${GPU_TIER:=standard}"
: "${SEED:=1}"
: "${WANDB_ENTITY:=graph_denoise_team}"
: "${WANDB_PROJECT:=discrete-diffusion}"
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

typeset -a cmd
cmd=(
  uv
  run
  tmgg-modal
  run
  tmgg-discrete-gen
  models/discrete@model=discrete_sbm_official
  +data=spectre_sbm
  # The base config carries inline synthetic-only keys (``graph_type``,
  # ``num_nodes``, ``num_graphs``, ``train_ratio``, ``val_ratio``,
  # ``graph_config``). They are absorbed by
  # ``SpectreSBMDataModule.__init__``'s ``**_metadata`` and re-exposed
  # at the data namespace for downstream interpolation (the wandb
  # logger reads ``${data.num_nodes}``). No ``~`` deletes needed.
  learning_rate=0.0002
  weight_decay=1e-12
  amsgrad=true
  seed="${SEED}"
  trainer.max_steps=550000
  # Validate every 10 000 steps. Upstream DiGress validates per-epoch
  # (~44k steps at our batch size); we deliberately validate ~4x more
  # often for faster training-time feedback.
  trainer.val_check_interval=10000
  model.eval_every_n_steps=10000
  model.noise_schedule.timesteps=1000
  +model.model.extra_features._target_=tmgg.models.digress.extra_features.ExtraFeatures
  +model.model.extra_features.extra_features_type=all
  # SPECTRE graphs reach n=187; cycle-feature counts need a ceiling
  # at-or-above the largest graph in the dataset.
  +model.model.extra_features.max_n_nodes=200
  allow_no_wandb=false
  wandb_entity="${WANDB_ENTITY}"
  wandb_project="${WANDB_PROJECT}"
  --gpu
  "${GPU_TIER}"
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

print -r -- "Launching upstream-style DiGress SBM run on Modal..."
printf ' %q' "${cmd[@]}"
print

run_prefixed "${cmd[@]}"
