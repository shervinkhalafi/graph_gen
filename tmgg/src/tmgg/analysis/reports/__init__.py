"""Concrete report generators.

Import all report modules here so ``@register_report`` decorators execute
when the analysis package loads.
"""

from tmgg.analysis.reports import eigenstructure  # noqa: F401
