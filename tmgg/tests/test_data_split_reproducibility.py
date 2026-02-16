"""Data splits must be reproducible given the same seed.

Rationale
---------
``data_module.py`` used Python's global ``random.shuffle``/``random.sample``
without passing a seed.  Two runs with the same config therefore produced
different train/val/test splits.  The fix introduces ``self._rng``, a
``random.Random(seed)`` instance, so that all partition sampling and matrix
shuffling become deterministic for a given seed.

Starting state: two freshly constructed ``GraphDataModule`` instances with
identical parameters (including ``seed``).

Invariants tested:
    - Identical seeds produce identical train/val/test adjacency matrices.
    - The SBM adjacency generation is also seeded, so the *content* of the
      matrices (not just the partition selection) matches.
"""

from __future__ import annotations

import torch


def test_sbm_splits_reproducible():
    """Two GraphDataModule instances with the same seed produce identical splits."""
    from tmgg.experiment_utils.data.data_module import GraphDataModule

    common = dict(
        dataset_name="sbm",
        dataset_config={
            "num_nodes": 12,
            "p_intra": 0.8,
            "p_inter": 0.1,
            "min_blocks": 2,
            "max_blocks": 3,
            "min_block_size": 2,
            "max_block_size": 8,
            "num_train_partitions": 5,
            "num_test_partitions": 5,
        },
        noise_levels=[0.1],
        batch_size=4,
        num_workers=0,
        seed=42,
        num_samples_per_graph=10,
    )

    dm1 = GraphDataModule(**common)
    dm1.prepare_data()
    dm1.setup()

    dm2 = GraphDataModule(**common)
    dm2.prepare_data()
    dm2.setup()

    # --- train ---
    assert dm1.train_adjacency_matrices is not None
    assert dm2.train_adjacency_matrices is not None
    assert len(dm1.train_adjacency_matrices) == len(dm2.train_adjacency_matrices)
    for i, (a, b) in enumerate(
        zip(dm1.train_adjacency_matrices, dm2.train_adjacency_matrices, strict=False)
    ):
        assert torch.equal(
            a, b
        ), f"Train matrix {i} differs between runs with same seed"

    # --- val ---
    assert dm1.val_adjacency_matrices is not None
    assert dm2.val_adjacency_matrices is not None
    assert len(dm1.val_adjacency_matrices) == len(dm2.val_adjacency_matrices)
    for i, (a, b) in enumerate(
        zip(dm1.val_adjacency_matrices, dm2.val_adjacency_matrices, strict=False)
    ):
        assert torch.equal(a, b), f"Val matrix {i} differs between runs with same seed"

    # --- test ---
    assert dm1.test_adjacency_matrices is not None
    assert dm2.test_adjacency_matrices is not None
    assert len(dm1.test_adjacency_matrices) == len(dm2.test_adjacency_matrices)
    for i, (a, b) in enumerate(
        zip(dm1.test_adjacency_matrices, dm2.test_adjacency_matrices, strict=False)
    ):
        assert torch.equal(a, b), f"Test matrix {i} differs between runs with same seed"


def test_different_seeds_produce_different_splits():
    """Different seeds should (with overwhelming probability) produce different splits."""
    from tmgg.experiment_utils.data.data_module import GraphDataModule

    base = dict(
        dataset_name="sbm",
        dataset_config={
            "num_nodes": 12,
            "p_intra": 0.8,
            "p_inter": 0.1,
            "min_blocks": 2,
            "max_blocks": 3,
            "min_block_size": 2,
            "max_block_size": 8,
            "num_train_partitions": 5,
            "num_test_partitions": 5,
        },
        noise_levels=[0.1],
        batch_size=4,
        num_workers=0,
        num_samples_per_graph=10,
    )

    dm1 = GraphDataModule(**base, seed=42)
    dm1.prepare_data()
    dm1.setup()

    dm2 = GraphDataModule(**base, seed=99)
    dm2.prepare_data()
    dm2.setup()

    assert dm1.train_adjacency_matrices is not None
    assert dm2.train_adjacency_matrices is not None

    # At least one matrix should differ (the SBM adjacency generation is random)
    any_differ = any(
        not torch.equal(a, b)
        for a, b in zip(
            dm1.train_adjacency_matrices, dm2.train_adjacency_matrices, strict=False
        )
    )
    assert (
        any_differ
    ), "Different seeds produced identical splits -- seeding is likely broken"
