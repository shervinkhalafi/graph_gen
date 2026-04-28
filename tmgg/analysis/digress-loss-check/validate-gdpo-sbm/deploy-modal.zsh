#!/usr/bin/env zsh
# Deploy the tmgg-validate-gdpo-sbm app to Modal. Re-run whenever
# validate.py or the checkpoint at .local-storage/... changes, since
# those are baked into the image at build time.

set -euo pipefail

script_dir=${0:A:h}
repo_root=${script_dir:A:h:h:h}  # validate-gdpo-sbm -> digress-loss-check -> analysis -> tmgg

cd "$repo_root"

exec uv run modal deploy analysis/digress-loss-check/validate-gdpo-sbm/modal_app.py "$@"
