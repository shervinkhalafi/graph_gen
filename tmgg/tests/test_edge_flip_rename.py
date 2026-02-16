"""Regression test: edge-flip noise uses correct names."""


def test_edge_flip_function_exists():
    from tmgg.experiment_utils.data.noise import add_edge_flip_noise

    assert callable(add_edge_flip_noise)


def test_edge_flip_generator_exists():
    from tmgg.experiment_utils.data.noise_generators import EdgeFlipNoiseGenerator

    assert EdgeFlipNoiseGenerator is not None


def test_factory_creates_edge_flip():
    from tmgg.experiment_utils.data.noise_generators import create_noise_generator

    gen = create_noise_generator("edge_flip")
    assert gen is not None


def test_factory_digress_still_works():
    """Existing 'digress' key must still work (backward compat).

    After Task 3b this will return the correct DigressNoiseGenerator,
    but for now it returns EdgeFlipNoiseGenerator.
    """
    from tmgg.experiment_utils.data.noise_generators import create_noise_generator

    gen = create_noise_generator("digress")
    assert gen is not None
