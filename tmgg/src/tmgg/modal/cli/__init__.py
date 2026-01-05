"""CLI tools for Modal experiment orchestration.

Provides
--------
generate_configs
    Generate experiment config JSONs from stage definitions.
    Usage: ``uv run python -m tmgg.modal.cli.generate_configs --stage stage1 --output-dir ./configs/``

launch_sweep
    Launch multiple experiments from a config directory.
    Usage: ``uv run python -m tmgg.modal.cli.launch_sweep --config-dir ./configs/ --gpu debug``

Workflow
--------
1. Generate configs: ``uv run python -m tmgg.modal.cli.generate_configs --stage stage1 --output-dir ./configs/stage1/``
2. Review generated configs (optional): inspect JSON files in output directory
3. Launch sweep: ``uv run python -m tmgg.modal.cli.launch_sweep --config-dir ./configs/stage1/ --gpu debug``
4. Monitor results: check Tigris storage and W&B

For single experiments, use run_single.py directly:
    ``doppler run -- uv run modal run --detach src/tmgg/modal/run_single.py --config ./config.json --gpu debug``
"""
