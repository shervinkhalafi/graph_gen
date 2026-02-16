"""Tests for shared sampling and noise schedule utilities.

Rationale
---------
``get_noise_schedule`` was previously inlined in both the Gaussian generative
lightning module (as a standalone function) and the DiGress denoising module
(as inline code in ``sample()``). Extracting it to a shared location prevents
drift between the two implementations and ensures consistent schedule
semantics across all modules.

The tests verify that each schedule type produces the right number of values,
stays within ``[0, 1]``, increases monotonically, and that unsupported types
raise ``ValueError``.
"""

import numpy as np
import pytest

from tmgg.experiment_utils.sampling import get_noise_schedule


def test_noise_schedule_linear():
    """Linear schedule should return evenly spaced values from 0 to 1."""
    schedule = get_noise_schedule("linear", 10)
    assert len(schedule) == 10
    assert schedule[0] < schedule[-1]  # increasing noise
    np.testing.assert_allclose(schedule, np.linspace(0, 1, 10))


def test_noise_schedule_cosine():
    """Cosine schedule should follow a half-cosine curve."""
    schedule = get_noise_schedule("cosine", 10)
    assert len(schedule) == 10
    assert schedule[0] < schedule[-1]


def test_noise_schedule_quadratic():
    """Quadratic schedule should follow a squared ramp."""
    schedule = get_noise_schedule("quadratic", 10)
    assert len(schedule) == 10
    assert schedule[0] < schedule[-1]


def test_noise_schedule_values_in_range():
    """All schedule values should lie in [0, 1]."""
    for stype in ("linear", "cosine", "quadratic"):
        schedule = get_noise_schedule(stype, 100)  # type: ignore[arg-type]
        assert all(
            0 <= v <= 1 for v in schedule
        ), f"{stype} schedule has out-of-range values"


def test_noise_schedule_monotonically_increasing():
    """All schedule types should produce monotonically non-decreasing values."""
    for stype in ("linear", "cosine", "quadratic"):
        schedule = get_noise_schedule(stype, 50)  # type: ignore[arg-type]
        diffs = np.diff(schedule)
        assert (diffs >= -1e-15).all(), (
            f"{stype} schedule is not monotonically increasing: "
            f"min diff = {diffs.min()}"
        )


def test_noise_schedule_boundary_values():
    """Schedules should start at 0 and end at 1."""
    for stype in ("linear", "cosine", "quadratic"):
        schedule = get_noise_schedule(stype, 100)  # type: ignore[arg-type]
        assert abs(schedule[0]) < 1e-10, f"{stype} schedule doesn't start at 0"
        assert abs(schedule[-1] - 1.0) < 1e-10, f"{stype} schedule doesn't end at 1"


def test_noise_schedule_unknown_raises():
    """Unknown schedule type should raise ValueError, not silently fall back."""
    with pytest.raises(ValueError, match="Unknown schedule"):
        get_noise_schedule("exponential", 10)  # type: ignore[arg-type]
