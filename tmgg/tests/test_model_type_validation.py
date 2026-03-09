"""Model type validation must use the factory registry, not hardcoded sets.

Rationale
---------
Validation is delegated to ``create_model`` which checks against
``ModelRegistry`` directly. This test verifies that every registered key
actually produces a working model.
"""

from tmgg.models.factory import ModelRegistry


def test_all_registry_keys_are_constructible():
    """Every key in ModelRegistry must produce a BaseModel without error.

    Guards against stale or broken registrations.
    """
    for name in ModelRegistry:
        model = ModelRegistry.create(name, {})
        assert model is not None, f"Factory for {name!r} returned None"
