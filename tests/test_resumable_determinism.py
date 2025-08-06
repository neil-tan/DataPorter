"""
Test resumable dataloader determinism and correctness.

These tests verify that ResumableDataLoader provides:
1. Deterministic batch ordering with the same seed
2. Correct state save/restore functionality
3. Identical behavior between continuous and pause/resume scenarios
4. Proper handling of epoch boundaries
"""

import pytest
import torch
from torch.utils.data import Dataset
import tempfile
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dataporter import ResumableDataLoader, create_resumable_dataloader


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


class TestResumableDataLoaderDeterminism:
    """Test determinism of ResumableDataLoader."""
    
    def test_same_seed_produces_same_order(self):
        """Test that dataloaders with the same seed produce identical batch order."""
        dataset = DeterministicDataset(size=20, seed=42)
        
        # Create two dataloaders with same seed
        dl1 = create_resumable_dataloader(
            dataset=dataset,
            batch_size=4,
            shuffle=True,
            seed=123,
            num_workers=0
        )
        
        dl2 = create_resumable_dataloader(
            dataset=dataset,
            batch_size=4,
            shuffle=True,
            seed=123,
            num_workers=0
        )
        
        # Collect all batches
        batches1 = [batch['id'].tolist() for batch in dl1]
        batches2 = [batch['id'].tolist() for batch in dl2]
        
        assert batches1 == batches2, "Same seed should produce identical batch order"
    
    def test_different_seeds_produce_different_order(self):
        """Test that dataloaders with different seeds produce different batch orders."""
        dataset = DeterministicDataset(size=20, seed=42)
        
        # Create two dataloaders with different seeds
        dl1 = create_resumable_dataloader(
            dataset=dataset,
            batch_size=4,
            shuffle=True,
            seed=123,
            num_workers=0
        )
        
        dl2 = create_resumable_dataloader(
            dataset=dataset,
            batch_size=4,
            shuffle=True,
            seed=456,
            num_workers=0
        )
        
        # Collect all batches
        batches1 = [batch['id'].tolist() for batch in dl1]
        batches2 = [batch['id'].tolist() for batch in dl2]
        
        assert batches1 != batches2, "Different seeds should produce different batch orders"
    
    def test_state_save_and_restore_preserves_position(self):
        """Test that saving and restoring state preserves the iteration position."""
        dataset = DeterministicDataset(size=20, seed=42)
        
        # Create dataloader and iterate partially
        dl = create_resumable_dataloader(
            dataset=dataset,
            batch_size=4,
            shuffle=True,
            seed=123,
            num_workers=0
        )
        
        # Get first 2 batches
        iter_dl = iter(dl)
        batch1 = next(iter_dl)
        batch2 = next(iter_dl)
        first_two = [batch1['id'].tolist(), batch2['id'].tolist()]
        
        # Save state
        state = dl.state_dict()
        
        # Get remaining batches
        remaining_original = [batch['id'].tolist() for batch in iter_dl]
        
        # Create new dataloader and restore state
        dl_restored = create_resumable_dataloader(
            dataset=dataset,
            batch_size=4,
            shuffle=True,
            seed=999,  # Different seed - should be overridden by state
            num_workers=0
        )
        
        dl_restored.load_state_dict(state)
        
        # Get remaining batches from restored dataloader
        remaining_restored = [batch['id'].tolist() for batch in dl_restored]
        
        assert remaining_original == remaining_restored, \
            "Restored dataloader should continue from the same position"
    
    def test_continuous_vs_pause_resume_identical(self):
        """Test that continuous iteration vs pause/resume produces identical results."""
        dataset = DeterministicDataset(size=20, seed=42)
        
        # Continuous run
        dl_continuous = create_resumable_dataloader(
            dataset=dataset,
            batch_size=4,
            shuffle=True,
            seed=123,
            num_workers=0
        )
        
        continuous_batches = [batch['id'].tolist() for batch in dl_continuous]
        
        # Pause/resume run
        dl_pausable = create_resumable_dataloader(
            dataset=dataset,
            batch_size=4,
            shuffle=True,
            seed=123,
            num_workers=0
        )
        
        pause_resume_batches = []
        
        # Get first 2 batches
        iter_pausable = iter(dl_pausable)
        for _ in range(2):
            batch = next(iter_pausable)
            pause_resume_batches.append(batch['id'].tolist())
        
        # Save state
        state = dl_pausable.state_dict()
        
        # Create new dataloader with different seed and restore
        dl_resumed = create_resumable_dataloader(
            dataset=dataset,
            batch_size=4,
            shuffle=True,
            seed=999,  # Different seed
            num_workers=0
        )
        
        dl_resumed.load_state_dict(state)
        
        # Get remaining batches
        for batch in dl_resumed:
            pause_resume_batches.append(batch['id'].tolist())
        
        assert continuous_batches == pause_resume_batches, \
            "Continuous and pause/resume runs should produce identical batch sequences"
    
    def test_epoch_boundary_handling(self):
        """Test that epoch boundaries are handled correctly during resume."""
        dataset = DeterministicDataset(size=10, seed=42)
        
        # Create dataloader that will iterate multiple epochs
        dl = create_resumable_dataloader(
            dataset=dataset,
            batch_size=3,
            shuffle=True,
            seed=123,
            num_workers=0,
            drop_last=False
        )
        
        # Iterate through first epoch (should be 4 batches: 10/3 = 3.33)
        all_batches = []
        for batch in dl:
            all_batches.append(batch['id'].tolist())
        
        # Save state after first epoch
        state = dl.state_dict()
        
        # Check that state correctly reflects position (4 batches in first epoch)
        assert state['batches_processed'] == 4
        assert state['epoch'] == 0  # Still in first epoch
        
        # Move to second epoch and get one more batch
        dl.set_epoch(1)
        for i, batch in enumerate(dl):
            if i >= 1:
                break
            all_batches.append(batch['id'].tolist())
        
        # Save state again
        state = dl.state_dict()
        # batches_processed is reset per epoch, so after 1 batch in epoch 1, it should be 1
        assert state['batches_processed'] == 1
        assert state['epoch'] == 1  # Now in second epoch
        
        # Create new dataloader and restore
        dl_restored = create_resumable_dataloader(
            dataset=dataset,
            batch_size=3,
            shuffle=True,
            seed=999,
            num_workers=0,
            drop_last=False
        )
        
        dl_restored.load_state_dict(state)
        
        # The restored dataloader should be in the second epoch
        # and should continue from where we left off
        restored_batches = []
        for batch in dl_restored:
            restored_batches.append(batch['id'].tolist())
            if len(all_batches) + len(restored_batches) >= 8:  # Get a few more batches
                break
        
        # Verify we got different batches (not starting from beginning)
        assert len(restored_batches) > 0
        assert restored_batches[0] not in all_batches[:3], \
            "Restored dataloader should not repeat batches from first epoch"
    
    def test_state_persistence_across_multiple_saves(self):
        """Test that state can be saved and restored multiple times correctly."""
        dataset = DeterministicDataset(size=20, seed=42)
        
        # Track all batches across multiple save/restore cycles
        all_batches = []
        
        # Initial dataloader
        dl = create_resumable_dataloader(
            dataset=dataset,
            batch_size=3,
            shuffle=True,
            seed=123,
            num_workers=0
        )
        
        # First segment: get 2 batches
        iter_dl = iter(dl)
        for _ in range(2):
            batch = next(iter_dl)
            all_batches.append(batch['id'].tolist())
        
        state1 = dl.state_dict()
        
        # Second segment: restore and get 2 more batches
        dl2 = create_resumable_dataloader(
            dataset=dataset,
            batch_size=3,
            shuffle=True,
            seed=456,
            num_workers=0
        )
        dl2.load_state_dict(state1)
        
        iter_dl2 = iter(dl2)
        for _ in range(2):
            batch = next(iter_dl2)
            all_batches.append(batch['id'].tolist())
        
        state2 = dl2.state_dict()
        
        # Third segment: restore and get remaining batches
        dl3 = create_resumable_dataloader(
            dataset=dataset,
            batch_size=3,
            shuffle=True,
            seed=789,
            num_workers=0
        )
        dl3.load_state_dict(state2)
        
        for batch in dl3:
            all_batches.append(batch['id'].tolist())
            if len(all_batches) >= 6:  # Stop after 6 total batches
                break
        
        # Compare with continuous run
        dl_continuous = create_resumable_dataloader(
            dataset=dataset,
            batch_size=3,
            shuffle=True,
            seed=123,  # Same as initial
            num_workers=0
        )
        
        continuous_batches = []
        for i, batch in enumerate(dl_continuous):
            if i >= 6:
                break
            continuous_batches.append(batch['id'].tolist())
        
        assert all_batches == continuous_batches, \
            "Multiple save/restore cycles should maintain batch order"
    
    def test_empty_dataset_handling(self):
        """Test that empty datasets are handled gracefully."""
        dataset = DeterministicDataset(size=0, seed=42)
        
        dl = create_resumable_dataloader(
            dataset=dataset,
            batch_size=4,
            shuffle=True,
            seed=123,
            num_workers=0
        )
        
        # Should not raise an error
        batches = list(dl)
        assert len(batches) == 0
        
        # State should still be saveable/loadable
        state = dl.state_dict()
        assert state['batches_processed'] == 0
        
        dl2 = create_resumable_dataloader(
            dataset=dataset,
            batch_size=4,
            shuffle=True,
            seed=456,
            num_workers=0
        )
        
        # Should not raise an error
        dl2.load_state_dict(state)
        batches2 = list(dl2)
        assert len(batches2) == 0
    
    def test_drop_last_behavior(self):
        """Test that drop_last behavior is consistent across resume."""
        dataset = DeterministicDataset(size=10, seed=42)
        
        # Test with drop_last=True
        dl_drop = create_resumable_dataloader(
            dataset=dataset,
            batch_size=3,
            shuffle=True,
            seed=123,
            num_workers=0,
            drop_last=True
        )
        
        drop_batches = [batch['id'].tolist() for batch in dl_drop]
        
        # Should drop the last incomplete batch (10 items / 3 = 3 complete batches)
        assert len(drop_batches) == 3
        assert all(len(batch) == 3 for batch in drop_batches)
        
        # Test resume with drop_last
        dl_drop_resume = create_resumable_dataloader(
            dataset=dataset,
            batch_size=3,
            shuffle=True,
            seed=123,
            num_workers=0,
            drop_last=True
        )
        
        # Get first batch then save
        iter_drop = iter(dl_drop_resume)
        first_batch = next(iter_drop)['id'].tolist()
        state = dl_drop_resume.state_dict()
        
        # Restore and get remaining
        dl_restored = create_resumable_dataloader(
            dataset=dataset,
            batch_size=3,
            shuffle=True,
            seed=999,
            num_workers=0,
            drop_last=True
        )
        dl_restored.load_state_dict(state)
        
        resume_batches = [first_batch]
        resume_batches.extend([batch['id'].tolist() for batch in dl_restored])
        
        assert drop_batches == resume_batches, \
            "drop_last behavior should be consistent across resume"


class TestResumableDataLoaderDistributed:
    """Test distributed training scenarios."""
    
    @pytest.mark.skipif(
        not torch.distributed.is_available(),
        reason="Distributed training not available"
    )
    def test_distributed_state_handling(self):
        """Test that distributed sampler state is handled correctly."""
        # This would require a distributed setup to test properly
        # For now, we just verify the API works
        dataset = DeterministicDataset(size=20, seed=42)
        
        dl = create_resumable_dataloader(
            dataset=dataset,
            batch_size=4,
            shuffle=True,
            seed=123,
            num_workers=0,
            distributed=False  # Explicitly disable for this test
        )
        
        state = dl.state_dict()
        assert 'distributed' in state
        assert state['distributed'] is False


class TestResumableDataLoaderEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_batch_size_change_warning(self):
        """Test that changing batch size after resume works but may affect reproducibility."""
        dataset = DeterministicDataset(size=20, seed=42)
        
        # Create dataloader with batch_size=4
        dl1 = create_resumable_dataloader(
            dataset=dataset,
            batch_size=4,
            shuffle=True,
            seed=123,
            num_workers=0
        )
        
        # Get 2 batches
        iter_dl1 = iter(dl1)
        next(iter_dl1)
        next(iter_dl1)
        
        state = dl1.state_dict()
        
        # Create new dataloader with different batch size
        dl2 = create_resumable_dataloader(
            dataset=dataset,
            batch_size=5,  # Different batch size
            shuffle=True,
            seed=123,
            num_workers=0
        )
        
        # This should work but may not maintain exact reproducibility
        dl2.load_state_dict(state)
        
        # Should be able to iterate
        remaining = list(dl2)
        assert len(remaining) > 0
    
    def test_num_workers_consistency(self):
        """Test that num_workers can be different between save and restore."""
        dataset = DeterministicDataset(size=20, seed=42)
        
        # Create dataloader with num_workers=0
        dl1 = create_resumable_dataloader(
            dataset=dataset,
            batch_size=4,
            shuffle=True,
            seed=123,
            num_workers=0
        )
        
        # Get some batches
        batches1 = []
        for i, batch in enumerate(dl1):
            if i >= 3:
                break
            batches1.append(batch['id'].tolist())
        
        state = dl1.state_dict()
        
        # Create new dataloader with num_workers=2
        dl2 = create_resumable_dataloader(
            dataset=dataset,
            batch_size=4,
            shuffle=True,
            seed=123,
            num_workers=2 if torch.multiprocessing.get_start_method() != 'spawn' else 0
        )
        
        # Should work fine
        dl2.load_state_dict(state)
        
        # Continue iteration
        batches2 = []
        for batch in dl2:
            batches2.append(batch['id'].tolist())
            if len(batches1) + len(batches2) >= 5:
                break
        
        # Verify we got new batches
        assert len(batches2) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])