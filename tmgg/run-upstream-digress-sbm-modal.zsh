#!/usr/bin/env zsh

set -euo pipefail

# Launch the current TMGG discrete generative path with a synthetic SBM setup
# chosen to approximate the upstream DiGress SBM training schedule in steps.

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
  +models/discrete@model=discrete_sbm_official
  learning_rate=0.0002
  weight_decay=1e-12
  amsgrad=true
  seed="${SEED}"
  data.num_graphs=200
  data.train_ratio=0.64
  data.val_ratio=0.16
  data.batch_size=12
  data.graph_config.p_intra=1.0
  data.graph_config.p_inter=0.0
  trainer.max_steps=550000
  trainer.val_check_interval=1100
  model.eval_every_n_steps=1100
  model.noise_schedule.timesteps=1000
  +model.model.extra_features._target_=tmgg.models.digress.extra_features.ExtraFeatures
  +model.model.extra_features.extra_features_type=all
  +model.model.extra_features.max_n_nodes=20
  +model.evaluator.p_intra=1.0
  +model.evaluator.p_inter=0.0
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
