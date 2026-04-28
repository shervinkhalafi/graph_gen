"""Client for the deployed ``tmgg-validate-gdpo-sbm`` Modal app.

Looks up a deployed function by name and either spawns an async call
(``spawn``) or fetches a previously-spawned call's result (``fetch``).
Unlike ``modal run`` this does NOT manage any app lifecycle: the
deployed app lives on Modal's side, so spawned calls keep executing
regardless of whether this client is still running.

Usage
-----

    # Deploy once (re-run after editing validate.py or bumping the
    # checkpoint, since they're baked into the image at build time):
    uv run modal deploy analysis/digress-loss-check/validate-gdpo-sbm/modal_app.py

    # Spawn a call (prints a FunctionCall id and exits):
    uv run python analysis/digress-loss-check/validate-gdpo-sbm/client.py spawn \
        --gpu a100 --num-samples 200

    # Fetch later (non-blocking unless --timeout > 0):
    uv run python analysis/digress-loss-check/validate-gdpo-sbm/client.py fetch \
        --call-id fc-... [--out-dir <path>] [--timeout 0]
"""

from __future__ import annotations

from pathlib import Path

import click
import modal

APP_NAME = "tmgg-validate-gdpo-sbm"
_THIS_FILE = Path(__file__).resolve()

_GPU_TO_FUNCTION_NAME = {
    "a10g": "validate_a10g",
    "a100": "validate_a100",
    "a100-40gb": "validate_a100",
    "t4": "validate_t4",
}


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--gpu", default="a10g", show_default=True)
@click.option("--num-samples", type=int, default=40, show_default=True)
@click.option("--batch-size", type=int, default=40, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
def spawn(gpu: str, num_samples: int, batch_size: int, seed: int) -> None:
    """Spawn a remote call against the deployed app and print the call id."""
    gpu_key = gpu.lower()
    if gpu_key not in _GPU_TO_FUNCTION_NAME:
        raise click.ClickException(
            f"unknown --gpu {gpu!r}; expected one of {sorted(_GPU_TO_FUNCTION_NAME)}"
        )
    fn_name = _GPU_TO_FUNCTION_NAME[gpu_key]

    click.echo(f"[client] looking up {APP_NAME}::{fn_name}...")
    fn = modal.Function.from_name(APP_NAME, fn_name)

    call = fn.spawn(num_samples=num_samples, batch_size=batch_size, seed=seed)
    click.echo(f"[client] spawned call_id={call.object_id}")
    click.echo("[client] fetch later with:")
    click.echo(
        f"  uv run python {_THIS_FILE} fetch --call-id {call.object_id}"
        " [--out-dir <path>] [--timeout 0]"
    )
    click.echo("[client] OR once the run finishes, pull outputs off the volume:")
    click.echo("  modal volume get tmgg-outputs /data/outputs/validate-gdpo-sbm/")


@cli.command()
@click.option("--call-id", required=True)
@click.option("--out-dir", default="", help="Where to write files locally.")
@click.option(
    "--timeout",
    type=float,
    default=0.0,
    show_default=True,
    help="Seconds to wait for completion. 0 = poll once (raise if pending).",
)
def fetch(call_id: str, out_dir: str, timeout: float) -> None:
    """Fetch the result of a previously spawned call and write its files."""
    call = modal.FunctionCall.from_id(call_id)
    click.echo(f"[client] fetching {call_id} (timeout={timeout}s)...")
    try:
        result = call.get(timeout=timeout)
    except TimeoutError as err:
        raise click.ClickException(
            f"call {call_id} is still pending — retry later or raise --timeout"
        ) from err
    except modal.exception.OutputExpiredError as err:
        raise click.ClickException(
            f"call {call_id} output expired (7-day retention). Re-spawn."
        ) from err

    if out_dir:
        target = Path(out_dir).resolve()
    else:
        target = _THIS_FILE.parent / "outputs" / f"modal-{result['stamp']}"
    target.mkdir(parents=True, exist_ok=True)

    for fname, blob in result["files"].items():
        if blob is None:
            click.echo(f"[client] WARN: {fname} was not produced by the container")
            continue
        (target / fname).write_bytes(blob)
        click.echo(f"[client] wrote {target / fname} ({len(blob)} bytes)")

    click.echo("[client] metrics:")
    for k, v in result["metrics"].items():
        click.echo(f"  {k}: {v}")
    click.echo(
        f"[client] container output dir (on volume): {result['output_dir']}"
        + f" — also retrievable via `modal volume get tmgg-outputs {result['output_dir']}`"
    )


if __name__ == "__main__":
    cli()
