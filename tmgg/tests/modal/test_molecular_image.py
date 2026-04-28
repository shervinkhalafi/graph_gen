"""Modal-marked: confirm molecular deps are present in the image.

Skipped on host pytest unless ``-m modal`` is passed. Runs as part
of Modal-side CI / deploy verification.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.modal]


def test_rdkit_imports() -> None:
    from rdkit import Chem  # noqa: F401


def test_fcd_torch_imports() -> None:
    from fcd_torch import FCD  # noqa: F401


def test_moses_imports() -> None:
    import moses  # noqa: F401


def test_guacamol_imports() -> None:
    import guacamol  # noqa: F401
