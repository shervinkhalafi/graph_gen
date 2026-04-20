#!/usr/bin/env bash
# Render docs/reports/2026-04-19-phase4-eigenvalue-study/figures-spec.typ to PDF
# via Typst. The .typ source embeds the generated Phase 4 figures via
# #image("figures/...") so no pre-processing is required.
#
# Usage:  scripts/render_phase4_spec_pdf.sh
#
# Requires: typst on PATH. Fails loudly if the binary or the figures directory
# is missing.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPORT_DIR="$REPO_ROOT/docs/reports/2026-04-19-phase4-eigenvalue-study"
SPEC_TYP="$REPORT_DIR/figures-spec.typ"
FIG_DIR="$REPORT_DIR/figures"
OUT_PDF="$REPORT_DIR/figures-spec.pdf"

command -v typst >/dev/null || { echo "typst not found on PATH" >&2; exit 1; }
[[ -f "$SPEC_TYP" ]] || { echo "missing spec: $SPEC_TYP" >&2; exit 1; }
[[ -d "$FIG_DIR" ]] || { echo "missing figures dir: $FIG_DIR — run scripts/plot_phase4_figures.py first" >&2; exit 1; }

# Typst resolves relative paths from the source file's directory, so compile
# from $REPORT_DIR to make `figures/...` image paths resolve.
typst compile --root "$REPO_ROOT" "$SPEC_TYP" "$OUT_PDF"

echo "wrote $OUT_PDF"
