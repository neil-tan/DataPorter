#!/usr/bin/env python3
"""Debug test to see what data is returned by ResumableDataLoader."""

import torch
from torch.utils.data import Dataset

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'external/DataPorter/src'))

from dataporter import create_resumable_dataloader


class SimpleDataset(Dataset):
    """Simple dataset for testing."""
    
    def __init__(self, size: int = 20):
        self.size = size
        self.data = list(range(size))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {'id': self.data[idx], 'value': idx * 2}


def test_debug():
    """Debug what ResumableDataLoader returns."""
    dataset = SimpleDataset(size=10)
    
    dl = create_resumable_dataloader(
        dataset=dataset,
        batch_size=2,
        shuffle=True,
        seed=123,
        num_workers=0
    )
    
    print("Getting first batch...")
    batch = next(iter(dl))
    print(f"Type of batch: {type(batch)}")
    print(f"Batch contents: {batch}")
    
    if isinstance(batch, dict):
        print(f"Keys in batch: {list(batch.keys())}")
        print(f"Type of batch['id']: {type(batch['id'])}")
        print(f"batch['id']: {batch['id']}")
    else:
        print(f"Type of first item: {type(batch[0])}")
        print(f"First item: {batch[0]}")


if __name__ == "__main__":
    test_debug()