#!/usr/bin/env python3
"""
Integration test for datamodule changes using actual experiment configuration.
Tests that the refactored datamodule works with real Hydra configs.
"""

import sys

sys.path.append("src")

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra


def test_config_loading():
    """Test that configs still load properly after refactoring."""
    print("Testing config loading...")

    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()

    config_path = "src/tmgg/exp_configs"

    try:
        with initialize(config_path=config_path, version_base="1.3"):
            # Test loading the base attention config
            cfg = compose(config_name="base_config_attention")

            print(f"  ✓ Loaded config with target: {cfg.data._target_}")
            print(f"  ✓ Dataset name: {cfg.data.dataset_name}")
            print(f"  ✓ Batch size: {cfg.data.batch_size}")
            print(f"  ✓ Num workers: {cfg.data.num_workers}")

            # Verify the inheritance worked
            assert cfg.data._target_ == "tmgg.experiment_utils.data.GraphDataModule"
            assert cfg.data.num_workers == 4  # Should come from base_dataloader
            assert cfg.data.pin_memory is True  # Should come from base_dataloader

            print("  ✓ Config inheritance working correctly")
            return True

    except Exception as e:
        print(f"  ✗ Config loading failed: {e}")
        return False


def test_datamodule_instantiation():
    """Test that we can instantiate the datamodule from config."""
    print("\nTesting datamodule instantiation from config...")

    GlobalHydra.instance().clear()

    try:
        with initialize(config_path="src/tmgg/exp_configs", version_base="1.3"):
            # Use the denoising-script-match config which has small fixed block_sizes
            cfg = compose(
                config_name="base_config_attention",
                overrides=["data=base/data/denoising-script-match"],
            )

            print(f"  ✓ Config data keys: {list(cfg.data.keys())}")
            print(f"  ✓ Dataset name: {cfg.data.dataset_name}")
            print(f"  ✓ Block sizes: {cfg.data.dataset_config.block_sizes}")

            # Instantiate the datamodule using Hydra
            from hydra.utils import instantiate

            datamodule = instantiate(cfg.data)

            print(f"  ✓ DataModule instantiated: {type(datamodule).__name__}")

            # Test basic functionality (but limit samples to avoid long runtime)
            # Modify the instantiated object for the test
            datamodule.num_samples_per_graph = 8  # Reduce for quick test
            datamodule.num_workers = 0  # No multiprocessing for test

            datamodule.prepare_data()
            datamodule.setup()

            print(
                f"  ✓ Setup completed. Train matrices: {len(datamodule.train_adjacency_matrices) if datamodule.train_adjacency_matrices else 0}"
            )

            # Test just that we can create a dataloader (don't iterate through all batches)
            train_loader = datamodule.train_dataloader()
            print(
                f"  ✓ Train dataloader created with {len(train_loader.dataset)} samples"
            )

            # Get just one batch to verify functionality
            batch = next(iter(train_loader))
            print(f"  ✓ Got sample batch with shape: {batch.shape}")

            print("  ✓ Full pipeline working correctly")

            return True

    except Exception as e:
        print(f"  ✗ Datamodule instantiation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run integration tests."""
    print("Running integration tests for datamodule refactoring...\n")

    tests = [test_config_loading, test_datamodule_instantiation]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print(f"\n{'=' * 50}")
    print(f"Integration tests completed: {passed}/{total} passed")

    if passed == total:
        print("✅ Integration tests passed! Datamodule works with experiment configs.")
        return 0
    else:
        print("❌ Some integration tests failed.")
        return 1


if __name__ == "__main__":
    exit(main())
