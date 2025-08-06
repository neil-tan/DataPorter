#!/usr/bin/env python3
"""Debug multi-epoch behavior."""

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


def test_multi_epoch():
    """Test multi-epoch behavior."""
    dataset = SimpleDataset(size=6)
    
    print("=== Testing Multi-Epoch Behavior ===")
    print(f"Dataset size: {len(dataset)}, batch_size: 2, batches per epoch: 3")
    
    # Create dataloader
    dl = create_resumable_dataloader(
        dataset=dataset,
        batch_size=2,
        shuffle=True,
        seed=123,
        num_workers=0
    )
    
    # Iterate through 5 batches (more than 1 epoch)
    all_batches = []
    print("\nIterating through 5 batches:")
    for i, batch in enumerate(dl):
        if i >= 5:
            break
        ids = batch['id'].tolist()
        all_batches.append(ids)
        print(f"Batch {i}: {ids}")
    
    # Save state
    state = dl.state_dict()
    print(f"\nState after 5 batches:")
    print(f"  batches_processed: {state['batches_processed']}")
    print(f"  epoch: {state['epoch']}")
    print(f"  Keys in state: {list(state.keys())}")
    
    # Create new dataloader and restore
    dl_restored = create_resumable_dataloader(
        dataset=dataset,
        batch_size=2,
        shuffle=True,
        seed=999,
        num_workers=0
    )
    
    dl_restored.load_state_dict(state)
    
    # Continue for 3 more batches
    restored_batches = []
    print("\nContinuing for 3 more batches after restore:")
    for i, batch in enumerate(dl_restored):
        if i >= 3:
            break
        ids = batch['id'].tolist()
        restored_batches.append(ids)
        print(f"Batch {i}: {ids}")
    
    # Show combined
    combined = all_batches + restored_batches
    print(f"\nCombined sequence (8 batches total): {combined}")
    
    # Now create continuous reference
    print("\n=== Continuous Reference ===")
    dl_continuous = create_resumable_dataloader(
        dataset=dataset,
        batch_size=2,
        shuffle=True,
        seed=123,
        num_workers=0
    )
    
    continuous_all = []
    for i, batch in enumerate(dl_continuous):
        if i >= 8:
            break
        ids = batch['id'].tolist()
        continuous_all.append(ids)
        print(f"Batch {i}: {ids}")
    
    print(f"\nContinuous sequence: {continuous_all}")
    
    # Compare
    print(f"\nDo they match? {combined == continuous_all}")
    if combined != continuous_all:
        print(f"Combined length: {len(combined)}")
        print(f"Continuous length: {len(continuous_all)}")


if __name__ == "__main__":
    test_multi_epoch()