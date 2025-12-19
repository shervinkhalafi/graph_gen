import sys

sys.path.append("src")

import torch

from tmgg.experiment_utils.data import GraphDataModule

config = {
    "dataset_name": "sbm",
    "dataset_config": {
        "block_sizes": [10, 5, 3, 2],
        "num_nodes": 20,
        "p_intra": 1.0,
        "q_inter": 0.0,
        "num_train_partitions": 1,
        "num_test_partitions": 1,
    },
    "num_samples_per_graph": 128,
    "batch_size": 32,
    "num_workers": 0,  # Use 0 for debugging
    "pin_memory": False,
    "val_split": 0.2,
    "test_split": 0.2,
    "noise_type": "digress",
    "noise_levels": [0.3],
}

dm = GraphDataModule(**config)
dm.prepare_data()
dm.setup("fit")

# Get a batch from the train dataloader
train_loader = dm.train_dataloader()
batch = next(iter(train_loader))

print(f"Batch type: {type(batch)}")
print(
    f"Batch shape: {batch.shape if hasattr(batch, 'shape') else 'No shape attribute'}"
)
print(
    f"Batch dtype: {batch.dtype if hasattr(batch, 'dtype') else 'No dtype attribute'}"
)
print(f"Is tensor: {isinstance(batch, torch.Tensor)}")
print(
    f"Batch content sample: {batch if not isinstance(batch, torch.Tensor) else 'Tensor OK'}"
)
