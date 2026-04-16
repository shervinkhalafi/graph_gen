"""Sanity checks for the GraphData field-name registry and loss dispatch."""

from typing import get_args

from tmgg.data.datasets.graph_data_fields import (
    FIELD_NAMES,
    GRAPHDATA_LOSS_KIND,
    FieldName,
)


def test_field_names_match_literal() -> None:
    """FIELD_NAMES contains exactly the four members of the FieldName Literal."""
    literal_members = set(get_args(FieldName))
    assert literal_members == {"X_class", "X_feat", "E_class", "E_feat"}
    assert set(FIELD_NAMES) == literal_members
    assert len(FIELD_NAMES) == 4


def test_loss_kind_keys_are_field_names() -> None:
    """Every key in GRAPHDATA_LOSS_KIND is a declared FieldName."""
    for key in GRAPHDATA_LOSS_KIND:
        assert key in FIELD_NAMES


def test_loss_kind_dispatch_by_suffix() -> None:
    """`_class` suffixes dispatch to cross-entropy, `_feat` suffixes to MSE."""
    for key, kind in GRAPHDATA_LOSS_KIND.items():
        if key.endswith("_class"):
            assert kind == "ce", f"{key} should map to 'ce', got {kind!r}"
        elif key.endswith("_feat"):
            assert kind == "mse", f"{key} should map to 'mse', got {kind!r}"
        else:
            raise AssertionError(f"Unexpected field name suffix: {key}")
