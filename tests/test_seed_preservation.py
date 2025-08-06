"""
Test that ResumableDataLoader correctly preserves the original seed during save/restore.

This test specifically verifies the fix for the seed preservation bug where
loading state into a dataloader created with a different seed would not
properly restore the original seed, causing batch order to diverge.
"""

import pytest
import torch
from torch.utils.data import Dataset

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

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


def test_seed_preservation_on_restore():
    """Test that the original seed is preserved when loading state."""
    dataset = SimpleDataset(size=20)
    
    # Create dataloader with seed=123
    dl_original = create_resumable_dataloader(
        dataset=dataset,
        batch_size=4,
        shuffle=True,
        seed=123,
        num_workers=0
    )
    
    # Collect first 2 batches
    original_batches = []
    iter_original = iter(dl_original)
    for _ in range(2):
        batch = next(iter_original)
        original_batches.append(batch['id'].tolist())
    
    # Save state
    state = dl_original.state_dict()
    
    # Verify the seed is in the state
    assert 'sampler_state' in state
    assert 'base_seed' in state['sampler_state']
    assert state['sampler_state']['base_seed'] == 123
    
    # Create new dataloader with DIFFERENT seed (999)
    dl_different_seed = create_resumable_dataloader(
        dataset=dataset,
        batch_size=4,
        shuffle=True,
        seed=999,  # Different seed!
        num_workers=0
    )
    
    # Before loading state, verify it has the different seed
    if hasattr(dl_different_seed, 'sampler') and hasattr(dl_different_seed.sampler, 'base_seed'):
        assert dl_different_seed.sampler.base_seed == 999
    
    # Load the saved state
    dl_different_seed.load_state_dict(state)
    
    # After loading state, verify the original seed was restored
    if hasattr(dl_different_seed, 'sampler') and hasattr(dl_different_seed.sampler, 'base_seed'):
        assert dl_different_seed.sampler.base_seed == 123, \
            "Original seed should be restored after loading state"
    
    # Get remaining batches
    remaining_batches = []
    for batch in dl_different_seed:
        remaining_batches.append(batch['id'].tolist())
    
    # Now create a continuous reference run with original seed
    dl_continuous = create_resumable_dataloader(
        dataset=dataset,
        batch_size=4,
        shuffle=True,
        seed=123,  # Same as original
        num_workers=0
    )
    
    continuous_batches = []
    for i, batch in enumerate(dl_continuous):
        continuous_batches.append(batch['id'].tolist())
        if i >= len(original_batches) + len(remaining_batches) - 1:
            break
    
    # Combine original and remaining batches
    combined_batches = original_batches + remaining_batches
    
    # They should match the continuous run
    assert combined_batches == continuous_batches[:len(combined_batches)], \
        "Batch order should be identical between continuous and pause/resume runs"


def test_seed_preservation_detailed():
    """Detailed test showing exact batch sequences."""
    dataset = SimpleDataset(size=10)
    
    print("\n=== Testing Seed Preservation ===")
    
    # Step 1: Create dataloader with seed=111
    dl1 = create_resumable_dataloader(
        dataset=dataset,
        batch_size=2,
        shuffle=True,
        seed=111,
        num_workers=0
    )
    
    # Get first 2 batches
    iter1 = iter(dl1)
    batch1 = next(iter1)
    batch2 = next(iter1)
    first_two_batches = [
        batch1['id'].tolist(),
        batch2['id'].tolist()
    ]
    print(f"First 2 batches with seed=111: {first_two_batches}")
    
    # Save state
    state = dl1.state_dict()
    print(f"Saved state seed: {state['sampler_state']['base_seed']}")
    
    # Get remaining batches from original
    remaining_original = []
    for batch in iter1:
        remaining_original.append(batch['id'].tolist())
    print(f"Remaining batches from original: {remaining_original}")
    
    # Step 2: Create new dataloader with DIFFERENT seed=999
    dl2 = create_resumable_dataloader(
        dataset=dataset,
        batch_size=2,
        shuffle=True,
        seed=999,  # Different seed!
        num_workers=0
    )
    
    print(f"\nCreated new dataloader with seed=999")
    
    # Show what batches it would produce without loading state
    dl2_no_restore = create_resumable_dataloader(
        dataset=dataset,
        batch_size=2,
        shuffle=True,
        seed=999,
        num_workers=0
    )
    wrong_batches = [batch['id'].tolist() for batch in dl2_no_restore]
    print(f"Batches with seed=999 (without restore): {wrong_batches[:3]}...")
    
    # Load state from original
    dl2.load_state_dict(state)
    print(f"\nLoaded state into new dataloader")
    
    # Check that seed was restored
    if hasattr(dl2, 'sampler') and hasattr(dl2.sampler, 'base_seed'):
        print(f"Sampler seed after restore: {dl2.sampler.base_seed}")
    
    # Get remaining batches
    remaining_restored = []
    for batch in dl2:
        remaining_restored.append(batch['id'].tolist())
    print(f"Remaining batches after restore: {remaining_restored}")
    
    # Verify they match
    assert remaining_original == remaining_restored, \
        "Remaining batches should be identical"
    
    print("\n✅ Seed preservation test passed!")


def test_multi_epoch_seed_preservation():
    """Test seed preservation across epoch boundaries."""
    dataset = SimpleDataset(size=6)  # Small dataset for multiple epochs
    
    # Create dataloader
    dl = create_resumable_dataloader(
        dataset=dataset,
        batch_size=2,
        shuffle=True,
        seed=123,
        num_workers=0
    )
    
    # Iterate through first epoch (3 batches)
    epoch1_batches = []
    for batch in dl:
        epoch1_batches.append(batch['id'].tolist())
    
    # Move to epoch 2
    dl.set_epoch(1)
    
    # Get 2 batches from epoch 2
    epoch2_batches = []
    for i, batch in enumerate(dl):
        if i >= 2:
            break
        epoch2_batches.append(batch['id'].tolist())
    
    # Save state (in middle of epoch 2)
    state = dl.state_dict()
    
    # Create new dataloader with different seed
    dl_restored = create_resumable_dataloader(
        dataset=dataset,
        batch_size=2,
        shuffle=True,
        seed=999,
        num_workers=0
    )
    
    # Load state
    dl_restored.load_state_dict(state)
    
    # Continue with last batch of epoch 2
    epoch2_continued = []
    for batch in dl_restored:
        epoch2_continued.append(batch['id'].tolist())
    
    # Move to epoch 3
    dl_restored.set_epoch(2)
    
    # Get batches from epoch 3
    epoch3_batches = []
    for i, batch in enumerate(dl_restored):
        if i >= 2:
            break
        epoch3_batches.append(batch['id'].tolist())
    
    # Create reference continuous run
    dl_continuous = create_resumable_dataloader(
        dataset=dataset,
        batch_size=2,
        shuffle=True,
        seed=123,
        num_workers=0
    )
    
    # Continuous run through same epochs
    continuous_all = []
    # Epoch 1
    for batch in dl_continuous:
        continuous_all.append(batch['id'].tolist())
    # Epoch 2
    dl_continuous.set_epoch(1)
    for batch in dl_continuous:
        continuous_all.append(batch['id'].tolist())
    # Epoch 3 (partial)
    dl_continuous.set_epoch(2)
    for i, batch in enumerate(dl_continuous):
        if i >= 2:
            break
        continuous_all.append(batch['id'].tolist())
    
    # Compare sequences
    combined = epoch1_batches + epoch2_batches + epoch2_continued + epoch3_batches
    assert combined == continuous_all, \
        f"Multi-epoch sequence should match continuous run\nCombined: {combined}\nContinuous: {continuous_all}"


if __name__ == "__main__":
    # Run the detailed test to show the behavior
    test_seed_preservation_detailed()
    
    # Run all tests
    pytest.main([__file__, "-v"])