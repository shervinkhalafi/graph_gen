"""Field identifiers and loss-kind dispatch for unified GraphData tensors.

Declares the canonical field names (node/edge class and feature streams) and
the loss kind each field dispatches to, shared across datasets, models, and
losses under the Wave 0 unified GraphData refactor.
"""

from typing import Final, Literal

FieldName = Literal["X_class", "X_feat", "E_class", "E_feat"]

FIELD_NAMES: Final[frozenset[FieldName]] = frozenset(
    {"X_class", "X_feat", "E_class", "E_feat"}
)

GRAPHDATA_LOSS_KIND: Final[dict[FieldName, Literal["ce", "mse"]]] = {
    "X_class": "ce",
    "X_feat": "mse",
    "E_class": "ce",
    "E_feat": "mse",
}
