#!/usr/bin/env python3
"""Simple test to verify ResumableDataLoader works with our fix."""

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


def test_basic_resumable():
    """Test basic resumable functionality."""
    dataset = SimpleDataset(size=10)
    
    print("=== Testing Basic Resumable Functionality ===")
    
    # Create dataloader with seed=42
    dl1 = create_resumable_dataloader(
        dataset=dataset,
        batch_size=3,
        shuffle=True,
        seed=42,
        num_workers=0
    )
    
    # Get first 2 batches
    batches1 = []
    iter1 = iter(dl1)
    for i in range(2):
        batch = next(iter1)
        batches1.append(batch['id'].tolist())
    print(f"First 2 batches: {batches1}")
    
    # Save state
    state = dl1.state_dict()
    print(f"Saved state - epoch: {state['epoch']}, batches: {state['batches_processed']}")
    
    # Get remaining batches
    remaining1 = []
    for batch in iter1:
        remaining1.append(batch['id'].tolist())
    print(f"Remaining batches: {remaining1}")
    
    # Create new dataloader with DIFFERENT seed
    dl2 = create_resumable_dataloader(
        dataset=dataset,
        batch_size=3,
        shuffle=True,
        seed=999,  # Different seed!
        num_workers=0
    )
    
    # Load state
    dl2.load_state_dict(state)
    
    # Get remaining batches from restored
    remaining2 = []
    for batch in dl2:
        remaining2.append(batch['id'].tolist())
    print(f"Remaining after restore: {remaining2}")
    
    # Compare
    if remaining1 == remaining2:
        print("\n✅ SUCCESS: Remaining batches match!")
    else:
        print("\n❌ FAIL: Remaining batches differ!")
        print(f"Expected: {remaining1}")
        print(f"Got: {remaining2}")
    
    # Now test continuous run
    print("\n=== Testing Continuous Run ===")
    dl_continuous = create_resumable_dataloader(
        dataset=dataset,
        batch_size=3,
        shuffle=True,
        seed=42,  # Same as original
        num_workers=0
    )
    
    continuous_batches = []
    for batch in dl_continuous:
        continuous_batches.append(batch['id'].tolist())
    print(f"Continuous run: {continuous_batches}")
    
    combined = batches1 + remaining1
    print(f"Combined pause/resume: {combined}")
    
    if continuous_batches == combined:
        print("\n✅ SUCCESS: Continuous and pause/resume match!")
    else:
        print("\n❌ FAIL: Continuous and pause/resume differ!")


if __name__ == "__main__":
    test_basic_resumable()