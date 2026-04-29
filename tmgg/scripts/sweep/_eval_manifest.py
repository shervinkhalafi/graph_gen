"""Backward-compatibility shim for the eval-manifest helper.

The canonical home is now ``tmgg.sweep._eval_manifest`` (inside the
package, so Modal containers can import it without sys.path hacks).
This shim re-exports every public symbol so existing call sites in
``scripts/sweep/{fetch_outcomes,watch_runs,reconcile_evals}.py`` and
host-side tests keep working unchanged.
"""

from tmgg.sweep._eval_manifest import (
    evals_completeness,
    evals_lag,
    expected_steps,
    latest_status_per_step,
    manifest_filename,
    read_manifest,
    resolve_manifest_dir,
    write_manifest_row,
)

__all__ = [
    "evals_completeness",
    "evals_lag",
    "expected_steps",
    "latest_status_per_step",
    "manifest_filename",
    "read_manifest",
    "resolve_manifest_dir",
    "write_manifest_row",
]
