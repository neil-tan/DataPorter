"""
Simplified test for seed preservation in ResumableDataLoader.
"""

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
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return idx  # Just return the index


def test_seed_preservation():
    """Test that the original seed is preserved when loading state."""
    dataset = SimpleDataset(size=10)
    
    print("=== Testing Seed Preservation ===")
    
    # Create dataloader with seed=123
    dl1 = create_resumable_dataloader(
        dataset=dataset,
        batch_size=2,
        shuffle=True,
        seed=123,
        num_workers=0
    )
    
    # Get first 2 batches
    iter1 = iter(dl1)
    batch1 = next(iter1)
    batch2 = next(iter1)
    first_two = [batch1.tolist(), batch2.tolist()]
    print(f"First 2 batches with seed=123: {first_two}")
    
    # Save state
    state = dl1.state_dict()
    print(f"Saved state - seed in state: {state['sampler_state']['base_seed']}")
    
    # Get remaining batches from original
    remaining_original = []
    for batch in iter1:
        remaining_original.append(batch.tolist())
    print(f"Remaining batches from original: {remaining_original}")
    
    # Create new dataloader with DIFFERENT seed=999
    dl2 = create_resumable_dataloader(
        dataset=dataset,
        batch_size=2,
        shuffle=True,
        seed=999,  # Different seed!
        num_workers=0
    )
    
    print(f"\nCreated new dataloader with seed=999")
    print(f"Sampler seed before load: {dl2.sampler.base_seed}")
    
    # Load state
    dl2.load_state_dict(state)
    print(f"Sampler seed after load: {dl2.sampler.base_seed}")
    
    # Get remaining batches
    remaining_restored = []
    for batch in dl2:
        remaining_restored.append(batch.tolist())
    print(f"Remaining batches after restore: {remaining_restored}")
    
    # Compare
    if remaining_original == remaining_restored:
        print("\n✅ Test PASSED: Remaining batches are identical!")
    else:
        print("\n❌ Test FAILED: Remaining batches differ!")
        print(f"Expected: {remaining_original}")
        print(f"Got: {remaining_restored}")
    
    # Also test full epoch comparison
    print("\n=== Full Epoch Comparison ===")
    
    # Continuous run
    dl_continuous = create_resumable_dataloader(
        dataset=dataset,
        batch_size=2,
        shuffle=True,
        seed=123,
        num_workers=0
    )
    
    continuous_batches = [batch.tolist() for batch in dl_continuous]
    print(f"Continuous run: {continuous_batches}")
    
    # Combined pause/resume
    combined = first_two + remaining_restored
    print(f"Pause/resume run: {combined}")
    
    if continuous_batches == combined:
        print("\n✅ Full epoch test PASSED!")
    else:
        print("\n❌ Full epoch test FAILED!")


if __name__ == "__main__":
    test_seed_preservation()