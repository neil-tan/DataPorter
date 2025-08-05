"""
Resumption strategies for ResumableDataLoader.

This module provides different strategies for handling dataloader resumption,
from simple batch counting to advanced memory-optimized approaches.
"""

import torch
from torch.utils.data import DataLoader, Sampler
from typing import Optional, Dict, Any, Iterator, Protocol
from abc import ABC, abstractmethod
from .samplers import ResumableSampler, ResumableDistributedSampler


class ResumptionStrategy(ABC):
    """
    Abstract base class for dataloader resumption strategies.
    
    Different strategies provide different trade-offs between simplicity,
    memory usage, and feature completeness.
    """
    
    def __init__(self):
        self.dataloader: Optional[DataLoader] = None
        
    def attach_dataloader(self, dataloader: DataLoader) -> None:
        """Attach this strategy to a dataloader."""
        self.dataloader = dataloader
        
    @abstractmethod
    def create_sampler(self, dataset, shuffle: bool = True, 
                      seed: Optional[int] = None) -> Optional[Sampler]:
        """Create an appropriate sampler for this strategy."""
        pass
        
    @abstractmethod
    def wrap_iterator(self, iterator: Iterator) -> Iterator:
        """Wrap the dataloader iterator to track progress."""
        pass
        
    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Return the state needed for resumption."""
        pass
        
    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load resumption state."""
        pass


class SimpleResumptionStrategy(ResumptionStrategy):
    """
    Simple resumption strategy with basic batch counting.
    
    Features:
    - Batch-level resumption (not sample-level)
    - Minimal memory overhead
    - No distributed support
    - Perfect for prototyping and small datasets
    
    This is the recommended starting point for most users.
    """
    
    def __init__(self):
        super().__init__()
        self.batches_seen = 0
        self.epoch = 0
        
    def create_sampler(self, dataset, shuffle: bool = True,
                      seed: Optional[int] = None) -> Optional[Sampler]:
        """Create a simple resumable sampler."""
        # For simple strategy, we can use ResumableSampler without optimizations
        return ResumableSampler(
            dataset, 
            shuffle=shuffle, 
            seed=seed if seed is not None else 42
        )
        
    def wrap_iterator(self, iterator: Iterator) -> Iterator:
        """Simple batch counting wrapper."""
        for batch in iterator:
            self.batches_seen += 1
            yield batch
            
    def state_dict(self) -> Dict[str, Any]:
        """Save simple state."""
        state = {
            'batches_seen': self.batches_seen,
            'epoch': self.epoch
        }
        
        # Include sampler state if available
        if (self.dataloader and hasattr(self.dataloader.sampler, 'state_dict')):
            state['sampler_state'] = self.dataloader.sampler.state_dict()
            
        return state
        
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load simple state."""
        self.batches_seen = state_dict.get('batches_seen', 0)
        self.epoch = state_dict.get('epoch', 0)
        
        # Load sampler state if available
        if (self.dataloader and 
            hasattr(self.dataloader.sampler, 'load_state_dict') and 
            'sampler_state' in state_dict):
            self.dataloader.sampler.load_state_dict(state_dict['sampler_state'])


class _AdvancedResumableIter:
    """Iterator wrapper for advanced strategy with precise tracking."""

    def __init__(self, base_iter: Iterator, strategy: 'AdvancedResumptionStrategy') -> None:
        self._iter = base_iter
        self._strategy = strategy

    def __iter__(self) -> '_AdvancedResumableIter':
        return self

    def __next__(self):
        batch = next(self._iter)
        self._strategy._batches_processed += 1
        return batch


class AdvancedResumptionStrategy(ResumptionStrategy):
    """
    Advanced resumption strategy with memory optimizations.
    
    Features:
    - Sample-level precision
    - Memory-optimized streaming
    - Multi-epoch handling
    - Production-ready performance
    
    This matches the current ResumableDataLoader implementation.
    """
    
    def __init__(self):
        super().__init__()
        self._batches_processed = 0
        self._epoch = 0
        self._distributed = False
        
    def create_sampler(self, dataset, shuffle: bool = True,
                      seed: Optional[int] = None) -> Optional[Sampler]:
        """Create memory-optimized sampler."""
        return ResumableSampler(
            dataset,
            shuffle=shuffle,
            seed=seed if seed is not None else 42
        )
        
    def wrap_iterator(self, iterator: Iterator) -> Iterator:
        """Use the current _ResumableIter wrapper."""
        return _AdvancedResumableIter(iterator, self)
        
    def state_dict(self) -> Dict[str, Any]:
        """Save full state including sampler."""
        state = {
            'batches_processed': self._batches_processed,
            'epoch': self._epoch,
            'distributed': self._distributed
        }
        
        if (self.dataloader and hasattr(self.dataloader.sampler, 'state_dict')):
            state['sampler_state'] = self.dataloader.sampler.state_dict()
            
        return state
        
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load full state with sample-level precision."""
        self._batches_processed = state_dict.get('batches_processed', 0)
        self._epoch = state_dict.get('epoch', 0)
        self._distributed = state_dict.get('distributed', False)
        
        if not self.dataloader:
            return
            
        # Calculate samples to skip
        batch_size = self.dataloader.batch_size
        samples_to_skip = self._batches_processed * batch_size
        dataset_size = len(self.dataloader.dataset)
        
        if dataset_size == 0:
            print(f"📊 ResumableDataLoader: Empty dataset, no resumption needed")
            return
        
        # Handle epoch overflow - advance logical epoch while maintaining shuffle consistency
        if samples_to_skip >= dataset_size:
            completed_epochs = samples_to_skip // dataset_size
            remaining_samples = samples_to_skip % dataset_size
            
            # Separate concerns: logical epoch tracking vs shuffle consistency
            sampler_epoch = self._epoch  # Keep original epoch for sampler shuffle consistency
            logical_epoch = self._epoch + completed_epochs  # Advance logical epoch tracking
            
            print(f"📊 ResumableDataLoader: Completed {completed_epochs} epoch(s), "
                  f"advancing to logical epoch {logical_epoch} at sample {remaining_samples}")
            
            samples_to_skip = remaining_samples
            # Advance logical epoch tracking for training progress
            self._epoch = logical_epoch
            self._batches_processed = remaining_samples // batch_size
            
            # Update sampler with original epoch to maintain shuffle consistency
            self._update_sampler_state(samples_to_skip, sampler_epoch, state_dict)
        else:
            print(f"📊 ResumableDataLoader: Resuming from batch {self._batches_processed} "
                  f"(skipping {samples_to_skip} samples)")
            
            # Update sampler with current epoch
            self._update_sampler_state(samples_to_skip, self._epoch, state_dict)
        
    def _update_sampler_state(self, samples_to_skip: int, epoch: int, 
                             state_dict: Dict[str, Any]) -> None:
        """Update the sampler's resumption state."""
        sampler = self.dataloader.sampler
        
        if isinstance(sampler, ResumableSampler):
            sampler.start_sample = samples_to_skip
            sampler.current_epoch = epoch
        elif hasattr(sampler, 'load_state_dict') and 'sampler_state' in state_dict:
            sampler.load_state_dict(state_dict['sampler_state'])


class DistributedResumptionStrategy(AdvancedResumptionStrategy):
    """
    Distributed training resumption strategy.
    
    Features:
    - All features of AdvancedResumptionStrategy
    - Multi-GPU synchronization
    - Rank-aware resumption
    - Distributed sampler support
    """
    
    def __init__(self):
        super().__init__()
        self._distributed = True
        
    def create_sampler(self, dataset, shuffle: bool = True,
                      seed: Optional[int] = None) -> Optional[Sampler]:
        """Create distributed resumable sampler."""
        # Require distributed training to be initialized for DistributedResumptionStrategy
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            raise RuntimeError(
                "DistributedResumptionStrategy requires distributed training to be initialized. "
                "Either initialize distributed training with torch.distributed.init_process_group() "
                "or use a different resumption strategy."
            )
        
        return ResumableDistributedSampler(
            dataset,
            shuffle=shuffle,
            seed=seed if seed is not None else 42,
            drop_last=False
        )
        
    def _update_sampler_state(self, samples_to_skip: int, epoch: int,
                             state_dict: Dict[str, Any]) -> None:
        """Update distributed sampler state."""
        sampler = self.dataloader.sampler
        
        if isinstance(sampler, ResumableDistributedSampler):
            sampler.start_sample = samples_to_skip
            sampler.current_epoch = epoch
            sampler.start_epoch = epoch
            sampler.set_epoch(epoch)
        else:
            super()._update_sampler_state(samples_to_skip, epoch, state_dict)