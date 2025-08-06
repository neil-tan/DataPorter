"""
Performance tests for ResumableDataLoader.

These tests verify that resumption functionality doesn't significantly
impact performance compared to standard PyTorch DataLoader.
"""

import time
import torch
from torch.utils.data import Dataset, DataLoader
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dataporter import create_resumable_dataloader


class LargeDataset(Dataset):
    """A larger dataset for performance testing."""
    
    def __init__(self, size: int = 10000):
        self.size = size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Simulate some data loading work
        data = torch.randn(224, 224, 3)  # Image-like data
        label = idx % 100
        return {'data': data, 'label': label, 'id': idx}


def measure_iteration_time(dataloader, num_batches: int = 100):
    """Measure time to iterate through a number of batches."""
    start_time = time.time()
    
    count = 0
    for batch in dataloader:
        count += 1
        if count >= num_batches:
            break
        # Simulate some processing
        _ = batch['data'].mean()
    
    end_time = time.time()
    return end_time - start_time, count


@pytest.mark.performance
def test_iteration_performance():
    """Test that ResumableDataLoader iteration performance is comparable to standard DataLoader."""
    dataset = LargeDataset(size=1000)
    batch_size = 32
    num_batches = 30
    
    # Standard PyTorch DataLoader
    standard_dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    # ResumableDataLoader
    resumable_dl = create_resumable_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        seed=42,
        pin_memory=False
    )
    
    # Warm up
    for _ in range(3):
        next(iter(standard_dl))
        next(iter(resumable_dl))
    
    # Measure standard DataLoader
    standard_time, _ = measure_iteration_time(standard_dl, num_batches)
    
    # Measure ResumableDataLoader
    resumable_time, _ = measure_iteration_time(resumable_dl, num_batches)
    
    # Calculate overhead
    overhead = (resumable_time - standard_time) / standard_time * 100
    
    print(f"\nPerformance Comparison (batch_size={batch_size}, num_batches={num_batches}):")
    print(f"Standard DataLoader: {standard_time:.3f}s")
    print(f"ResumableDataLoader: {resumable_time:.3f}s")
    print(f"Overhead: {overhead:.1f}%")
    
    # Assert overhead is reasonable (less than 10%)
    assert overhead < 10, f"ResumableDataLoader overhead too high: {overhead:.1f}%"


@pytest.mark.performance
def test_state_save_performance():
    """Test performance of state saving operations."""
    dataset = LargeDataset(size=10000)
    
    dl = create_resumable_dataloader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        seed=42
    )
    
    # Iterate through some batches
    for i, _ in enumerate(dl):
        if i >= 100:
            break
    
    # Measure state save time
    start_time = time.time()
    state = dl.state_dict()
    save_time = time.time() - start_time
    
    print(f"\nState save time: {save_time*1000:.2f}ms")
    
    # State saving should be very fast (< 10ms)
    assert save_time < 0.01, f"State saving too slow: {save_time*1000:.2f}ms"
    
    # Measure state restore time
    dl2 = create_resumable_dataloader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        seed=999
    )
    
    start_time = time.time()
    dl2.load_state_dict(state)
    restore_time = time.time() - start_time
    
    print(f"State restore time: {restore_time*1000:.2f}ms")
    
    # State restoring should be very fast (< 10ms)
    assert restore_time < 0.01, f"State restoring too slow: {restore_time*1000:.2f}ms"


@pytest.mark.performance
def test_resume_performance():
    """Test performance of resuming from different positions."""
    dataset = LargeDataset(size=10000)
    batch_size = 32
    
    resume_positions = [10, 100, 1000]
    
    print("\nResume Performance Test:")
    
    for resume_pos in resume_positions:
        # Create and iterate to resume position
        dl = create_resumable_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            seed=42
        )
        
        for i, _ in enumerate(dl):
            if i >= resume_pos:
                break
        
        state = dl.state_dict()
        
        # Measure resume time
        dl_resume = create_resumable_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            seed=999
        )
        
        start_time = time.time()
        dl_resume.load_state_dict(state)
        
        # Get first batch after resume
        next(iter(dl_resume))
        
        resume_time = time.time() - start_time
        
        print(f"Resume from position {resume_pos}: {resume_time*1000:.2f}ms")
        
        # Resume should be fast regardless of position
        assert resume_time < 0.1, f"Resume too slow from position {resume_pos}: {resume_time*1000:.2f}ms"


def test_memory_efficiency():
    """Test that ResumableDataLoader doesn't leak memory during iteration."""
    import gc
    import tracemalloc
    
    dataset = LargeDataset(size=1000)
    
    # Start memory tracking
    tracemalloc.start()
    
    # Create dataloader and iterate
    dl = create_resumable_dataloader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        seed=42
    )
    
    # Take initial snapshot
    gc.collect()
    snapshot1 = tracemalloc.take_snapshot()
    
    # Iterate through dataset multiple times
    for epoch in range(3):
        dl.set_epoch(epoch)
        for i, batch in enumerate(dl):
            if i >= 20:  # Process 20 batches per epoch
                break
            # Simulate some work
            _ = batch['data'].mean()
    
    # Take final snapshot
    gc.collect()
    snapshot2 = tracemalloc.take_snapshot()
    
    # Calculate memory difference
    stats = snapshot2.compare_to(snapshot1, 'lineno')
    
    # Find statistics related to our code
    total_diff = sum(stat.size_diff for stat in stats if stat.size_diff > 0)
    
    # Convert to MB
    memory_increase_mb = total_diff / 1024 / 1024
    
    print(f"\nMemory increase after 3 epochs: {memory_increase_mb:.2f} MB")
    
    # Memory increase should be minimal (< 10 MB)
    assert memory_increase_mb < 10, f"Memory leak detected: {memory_increase_mb:.2f} MB increase"
    
    tracemalloc.stop()


if __name__ == "__main__":
    # Run performance tests
    test_iteration_performance()
    test_state_save_performance()
    test_resume_performance()
    test_memory_efficiency()
    
    print("\n✅ All performance tests passed!")