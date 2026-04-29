"""Smoke-test the smoke launcher itself.

Two cheap structural checks on ``scripts/sweep/smoke_async_eval.zsh``
and the committed schedule YAML it consumes:

1. ``--help`` exits 0 and the usage banner mentions both "smoke" and
   "async" (the script's whole reason for existing).
2. The embedded round.yaml heredoc parses as YAML and has the keys the
   downstream ``scripts.sweep.launch_round`` consumer requires
   (``round`` int, ``launches`` list with one entry carrying ``dataset``,
   ``async_eval: true``, an ``overrides`` mapping). Catches drift between
   the inline template and the launcher's contract.

Test rationale: this script is not unit-testable end-to-end without
a Modal account and W&B credentials, so we validate the static
artefacts (help text, embedded YAML, committed schedule YAML) rather
than the launch flow.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "sweep" / "smoke_async_eval.zsh"
SCHEDULE_YAML = (
    REPO_ROOT
    / "docs"
    / "experiments"
    / "sweep"
    / "smallest-config-2026-04-29"
    / "smoke_eval_schedule.yaml"
)


def _extract_inline_round_yaml(script_text: str) -> str:
    """Grep the ``cat >...<<'EOF' ... EOF`` block out of the zsh source.

    The launcher writes a fresh round.yaml under ``/tmp`` from a quoted
    heredoc; we recover that body to verify it parses as YAML. Quoting
    the heredoc terminator (``<<'EOF'``) means zsh does not interpolate
    the contents, so what we read here is byte-identical to what the
    script writes at runtime.
    """
    match = re.search(
        r"cat >\"?\$\{round_yaml\}\"? <<'EOF'\n(?P<body>.*?)\nEOF\n",
        script_text,
        re.DOTALL,
    )
    if match is None:
        raise AssertionError(
            "Could not locate the round.yaml heredoc inside "
            "smoke_async_eval.zsh — has the template moved?"
        )
    return match.group("body")


def test_smoke_zsh_help_succeeds() -> None:
    """``--help`` returns 0 and mentions both 'smoke' and 'async'."""
    proc = subprocess.run(
        ["zsh", str(SMOKE_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0, (
        f"smoke_async_eval.zsh --help exited {proc.returncode}; "
        f"stderr={proc.stderr!r}"
    )
    out_lower = proc.stdout.lower()
    assert "smoke" in out_lower, "help banner should mention 'smoke'"
    assert "async" in out_lower, "help banner should mention 'async'"


def test_smoke_round_yaml_template_is_valid_yaml() -> None:
    """The embedded round.yaml heredoc parses as valid YAML.

    Required keys (consumed by ``scripts.sweep.launch_round``):
      - ``round`` (int)
      - ``launches`` (non-empty list)
      - first launch has ``dataset``, ``async_eval: true``, ``overrides``
    """
    body = _extract_inline_round_yaml(SMOKE_SCRIPT.read_text())
    payload = yaml.safe_load(body)
    assert isinstance(payload, dict), "round.yaml template must be a mapping"
    assert "round" in payload and isinstance(
        payload["round"], int
    ), "round.yaml template missing integer 'round' key"
    assert "launches" in payload and isinstance(
        payload["launches"], list
    ), "round.yaml template missing 'launches' list"
    assert (
        len(payload["launches"]) >= 1
    ), "round.yaml template must have at least one launch entry"
    first = payload["launches"][0]
    assert (
        first.get("dataset") == "spectre_sbm"
    ), "first launch entry must target spectre_sbm"
    assert (
        first.get("async_eval") is True
    ), "first launch entry must set async_eval: true"
    assert isinstance(
        first.get("overrides"), dict
    ), "first launch entry must carry an 'overrides' mapping"


def test_smoke_schedule_yaml_is_valid() -> None:
    """The committed smoke schedule yaml parses and has the keys
    ``scripts.sweep.launch_round.load_async_eval_schedule`` consumes.
    """
    payload = yaml.safe_load(SCHEDULE_YAML.read_text())
    assert isinstance(payload, dict)
    assert payload.get("dataset") == "spectre_sbm"
    schedule = payload.get("schedule")
    assert (
        isinstance(schedule, list) and len(schedule) == 2
    ), "smoke schedule must be a 2-element list"
    assert all(isinstance(s, int) for s in schedule), "schedule entries must be ints"
    assert schedule == [
        200,
        800,
    ], f"smoke schedule must be [200, 800], got {schedule!r}"


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_smoke_zsh_help_aliases(flag: str) -> None:
    """Both ``--help`` and ``-h`` short-circuit to usage."""
    proc = subprocess.run(
        ["zsh", str(SMOKE_SCRIPT), flag],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0
    assert "Usage:" in proc.stdout
