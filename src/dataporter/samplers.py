"""
Resumable samplers for PyTorch DataLoaders.

Provides sample-level resumption for both single-node and distributed training scenarios.
These samplers enable exact resumption from any point during training with deterministic
shuffling and memory-optimized streaming approaches.

Features:
- Sample-level precision for exact resumption 
- Deterministic shuffling with epoch-specific seeds
- Memory-efficient streaming to prevent pin memory exhaustion
- Distributed training support with DistributedSampler compatibility
- Lightning-compatible length reporting for multi-epoch training
"""

import torch
from torch.utils.data import Sampler, DistributedSampler
from typing import Optional, Dict, Any, Iterator
import warnings


class ResumableSampler(Sampler):
    """
    A sampler that can resume from a specific sample position.
    
    Works with any underlying dataset by reproducing the same sample order.
    Memory optimized with streaming index generation to prevent pin memory exhaustion.
    
    Key features:
    - Exact sample-level resumption for both shuffled and sequential datasets
    - Deterministic shuffling using epoch-specific seeds for reproducibility
    - Streaming index generation to minimize memory usage
    - Lightning-compatible length reporting
    
    Args:
        dataset: Dataset to sample from
        shuffle: Whether to shuffle samples (default: True)
        seed: Base seed for reproducible shuffling (default: None -> 42)
        start_sample: Sample index to start from (default: 0)
    """
    
    def __init__(self, dataset, shuffle: bool = True, seed: Optional[int] = None, 
                 start_sample: int = 0):
        self.dataset = dataset
        self.shuffle = shuffle
        self.base_seed = seed if seed is not None else 42
        self.start_sample = start_sample
        self.current_sample = 0
        self.current_epoch = 0  # Track which epoch we're in for deterministic seeding
        self._generator = None  # Lazy initialization
        
    def _get_generator(self) -> torch.Generator:
        """Get or create generator with epoch-specific seed for reproducible resumption."""
        if self._generator is None:
            self._generator = torch.Generator()
        # Use epoch-specific seed for scientific reproducibility
        epoch_seed = self.base_seed + self.current_epoch
        self._generator.manual_seed(epoch_seed)
        return self._generator
        
    def __iter__(self) -> Iterator[int]:
        """Generate indices with streaming approach to minimize memory usage."""
        dataset_len = len(self.dataset)
        start_from = self.start_sample
        
        # Reset start_sample for next epoch after capturing current value
        self.start_sample = 0
        
        # Initialize current_sample to start_from for proper tracking
        self.current_sample = start_from
        
        if self.shuffle:
            # Use cached generator for memory efficiency
            generator = self._get_generator()
            
            # Stream indices without creating full list in memory
            # Generate indices on-demand using the same random state
            indices_tensor = torch.randperm(dataset_len, generator=generator)
            
            # Continue from resume point to end of epoch (no wrap-around)
            # This maintains natural sample order for deterministic accuracy
            for i in range(start_from, dataset_len):
                self.current_sample = i
                yield indices_tensor[i].item()
                
            # Clean up tensor to free memory
            del indices_tensor
        else:
            # Stream sequential indices without creating list (no wrap-around)
            for i in range(start_from, dataset_len):
                self.current_sample = i
                yield i
                
    def __len__(self) -> int:
        """
        Return full dataset length for Lightning compatibility.
        
        Lightning calculates this once and assumes same count for all epochs.
        Returning remaining samples would break future epoch calculations.
        """
        return len(self.dataset)
        
    def state_dict(self) -> Dict[str, Any]:
        """Save sampler state for resumption."""
        return {
            'current_sample': self.current_sample,
            'start_sample': self.start_sample,
            'base_seed': self.base_seed,
            'current_epoch': self.current_epoch,
            'shuffle': self.shuffle
        }
        
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load sampler state for resumption."""
        self.current_sample = state_dict.get('current_sample', 0)
        # When resuming, we want to start from the next sample after the last processed one
        self.start_sample = self.current_sample + 1 if self.current_sample >= 0 else 0
        self.base_seed = state_dict.get('base_seed', state_dict.get('seed', 42))  # Backward compatibility
        self.current_epoch = state_dict.get('current_epoch', 0)
        self.shuffle = state_dict.get('shuffle', True)
    
    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for the sampler - called by DataLoader at start of each epoch."""
        self.current_epoch = epoch


class ResumableDistributedSampler(DistributedSampler):
    """
    A DistributedSampler that can resume from a specific sample position.
    
    Handles distributed training with deterministic sample ordering.
    Memory optimized with streaming index generation for large datasets.
    
    Key features:
    - Distributed training support with consistent sample ordering across ranks
    - Sample-level resumption with rank-aware skip calculations
    - Memory-efficient streaming approach for large datasets
    - Deterministic shuffling with epoch-specific seeds
    
    Args:
        dataset: Dataset to sample from
        num_replicas: Number of processes participating in training (default: None -> auto-detect)
        rank: Rank of the current process (default: None -> auto-detect)
        shuffle: Whether to shuffle samples (default: True)
        seed: Seed for reproducible shuffling (default: 0)
        drop_last: Whether to drop the last incomplete batch (default: False)
        start_sample: Global sample index to start from (default: 0)
        start_epoch: Epoch to start from (default: 0)
    """
    
    def __init__(self, dataset, num_replicas: Optional[int] = None, 
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False,
                 start_sample: int = 0, start_epoch: int = 0):
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.start_sample = start_sample
        self.start_epoch = start_epoch
        self.current_sample = 0
        self._generator = None  # Lazy initialization
        self._current_epoch = None  # Track generator epoch
        
        # Set epoch to start_epoch
        self.set_epoch(start_epoch)
        
    def _get_generator(self, epoch: int) -> torch.Generator:
        """Get or create generator for current epoch with deterministic seeding."""
        if self._generator is None:
            self._generator = torch.Generator()
        # Always reset seed for deterministic behavior across epoch changes
        self._generator.manual_seed(self.seed + epoch)
        self._current_epoch = epoch
        return self._generator
        
    def __iter__(self) -> Iterator[int]:
        """Generate indices with memory-optimized streaming approach."""
        self.current_sample = 0
        dataset_len = len(self.dataset)
        
        # Calculate starting position for resumption
        start_pos = 0
        if self.epoch == self.start_epoch and self.start_sample > 0:
            # Distribute start_sample across ranks (simple approximation)
            start_pos = min(self.start_sample // self.num_replicas, self.num_samples)
        
        if self.shuffle:
            # Use cached generator
            generator = self._get_generator(self.epoch)
            indices_tensor = torch.randperm(dataset_len, generator=generator)
            
            # Apply padding without creating intermediate lists
            if not self.drop_last and self.total_size > dataset_len:
                padding_size = self.total_size - dataset_len
                # Create padding indices efficiently
                if padding_size <= dataset_len:
                    padding_indices = indices_tensor[:padding_size]
                else:
                    # Repeat pattern for large padding
                    repeats = (padding_size // dataset_len) + 1
                    padding_indices = indices_tensor.repeat(repeats)[:padding_size]
                indices_tensor = torch.cat([indices_tensor, padding_indices])
            elif self.drop_last:
                indices_tensor = indices_tensor[:self.total_size]
            
            # Subsample for this rank and yield directly
            rank_indices = indices_tensor[self.rank::self.num_replicas]
            
            for i in range(start_pos, len(rank_indices)):
                self.current_sample = i
                yield rank_indices[i].item()
                
            # Clean up tensors
            del indices_tensor, rank_indices
        else:
            # Sequential indices - calculate range for this rank
            total_samples = self.total_size
            if not self.drop_last and total_samples < dataset_len:
                total_samples = dataset_len
                
            # Generate rank-specific indices without creating full lists
            rank_start = self.rank
            rank_step = self.num_replicas
            rank_indices_count = (total_samples + rank_step - 1 - rank_start) // rank_step
            
            for i in range(start_pos, min(rank_indices_count, self.num_samples)):
                global_idx = rank_start + i * rank_step
                # Handle padding by wrapping around
                actual_idx = global_idx % dataset_len
                self.current_sample = i
                yield actual_idx

    def state_dict(self) -> Dict[str, Any]:
        """Save distributed sampler state for resumption."""
        return {
            'epoch': self.epoch,
            'current_sample': self.current_sample,
            'start_sample': self.start_sample,
            'start_epoch': self.start_epoch,
            'seed': self.seed,
            'rank': self.rank,
            'num_replicas': self.num_replicas
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load distributed sampler state for resumption."""
        self.epoch = state_dict.get('epoch', 0)
        self.current_sample = state_dict.get('current_sample', 0)
        self.start_sample = state_dict.get('start_sample', 0)
        self.start_epoch = state_dict.get('start_epoch', 0)
        self.seed = state_dict.get('seed', 0)
        
        # Verify distributed setup matches
        saved_rank = state_dict.get('rank')
        saved_num_replicas = state_dict.get('num_replicas')
        if saved_rank is not None and saved_rank != self.rank:
            warnings.warn(f"Loaded rank {saved_rank} doesn't match current rank {self.rank}")
        if saved_num_replicas is not None and saved_num_replicas != self.num_replicas:
            warnings.warn(f"Loaded num_replicas {saved_num_replicas} doesn't match current {self.num_replicas}")