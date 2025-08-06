#!/usr/bin/env python3
"""Debug epoch boundary behavior."""

import torch
from torch.utils.data import Dataset

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'external/DataPorter/src'))

from dataporter import create_resumable_dataloader


class DeterministicDataset(Dataset):
    """A simple dataset with deterministic data for testing."""
    
    def __init__(self, size: int = 100, seed: int = 42):
        self.size = size
        self.seed = seed
        
        # Pre-generate all data deterministically
        self.data = []
        for i in range(size):
            torch.manual_seed(seed + i)
            self.data.append({
                'id': i,
                'data': torch.randn(10),
                'label': torch.randint(0, 10, (1,)).item()
            })
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx]


def test_epoch_boundary():
    """Test epoch boundary handling."""
    dataset = DeterministicDataset(size=10, seed=42)
    
    print("=== Testing Epoch Boundary ===")
    print(f"Dataset size: {len(dataset)}, batch_size: 3")
    print(f"Expected batches per epoch: 10/3 = 3.33 → 4 batches (with drop_last=False)")
    
    # Create dataloader
    dl = create_resumable_dataloader(
        dataset=dataset,
        batch_size=3,
        shuffle=True,
        seed=123,
        num_workers=0,
        drop_last=False
    )
    
    # Try to get 5 batches
    all_batches = []
    batches_to_get = 5
    
    print(f"\nTrying to get {batches_to_get} batches:")
    for i, batch in enumerate(dl):
        if i >= batches_to_get:
            break
        ids = batch['id'].tolist()
        all_batches.append(ids)
        print(f"Batch {i}: {ids} (size: {len(ids)})")
    
    # Save state
    state = dl.state_dict()
    print(f"\nState after iteration:")
    print(f"  batches_processed: {state['batches_processed']}")
    print(f"  epoch: {state['epoch']}")
    print(f"  Total batches retrieved: {len(all_batches)}")
    
    # Try manual epoch advancement
    print("\n=== Manual Epoch Advancement ===")
    dl2 = create_resumable_dataloader(
        dataset=dataset,
        batch_size=3,
        shuffle=True,
        seed=123,
        num_workers=0,
        drop_last=False
    )
    
    all_batches2 = []
    # Get all batches from epoch 0
    print("Epoch 0:")
    for i, batch in enumerate(dl2):
        ids = batch['id'].tolist()
        all_batches2.append(ids)
        print(f"  Batch {i}: {ids} (size: {len(ids)})")
    
    # Move to epoch 1
    dl2.set_epoch(1)
    print("\nEpoch 1 (after set_epoch(1)):")
    for i, batch in enumerate(dl2):
        if len(all_batches2) >= 5:
            break
        ids = batch['id'].tolist()
        all_batches2.append(ids)
        print(f"  Batch {i}: {ids} (size: {len(ids)})")
    
    print(f"\nTotal batches from manual epoch advancement: {len(all_batches2)}")
    
    # Save state
    state2 = dl2.state_dict()
    print(f"\nState after manual epoch advancement:")
    print(f"  batches_processed: {state2['batches_processed']}")
    print(f"  epoch: {state2['epoch']}")


if __name__ == "__main__":
    test_epoch_boundary()