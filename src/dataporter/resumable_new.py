"""
ResumableDataLoader implementation with strategy pattern for flexible resumption.

This refactored version provides the same functionality as the original but with
a cleaner architecture that separates concerns and makes simple use cases simpler.

Backward compatibility is maintained through the factory function and default
strategy selection.
"""

import torch
from torch.utils.data import DataLoader, Sampler
from typing import Optional, Dict, Any, Iterator
from .samplers import ResumableSampler, ResumableDistributedSampler
from .strategies import (
    ResumptionStrategy, SimpleResumptionStrategy, 
    AdvancedResumptionStrategy, DistributedResumptionStrategy
)


class ResumableDataLoader(DataLoader):
    """
    A DataLoader that can save and load its state for resuming training.
    
    This refactored version uses a strategy pattern to provide different levels
    of functionality based on user needs, from simple batch counting to 
    advanced memory-optimized resumption.
    
    Features:
    - Pluggable resumption strategies
    - Simple by default, powerful when needed
    - Backward compatible with existing code
    - Lightning-compatible state management
    
    Args:
        dataset: Dataset to load from
        batch_size: Number of samples per batch (default: 1)
        shuffle: Whether to shuffle samples (default: None -> True if no sampler)
        sampler: Custom sampler (default: None -> strategy will create one)
        resumption_strategy: Strategy for handling resumption (default: None -> auto)
        **kwargs: Additional arguments passed to parent DataLoader
    """
    
    def __init__(self, dataset, batch_size: int = 1, shuffle: bool = None,
                 sampler: Optional[Sampler] = None, batch_sampler=None,
                 num_workers: int = 0, collate_fn=None, pin_memory: bool = False,
                 drop_last: bool = False, timeout: float = 0,
                 worker_init_fn=None, multiprocessing_context=None,
                 generator=None, prefetch_factor: int = 2,
                 persistent_workers: bool = False,
                 # New strategy-based parameters
                 resumption_strategy: Optional[ResumptionStrategy] = None,
                 # Legacy parameters for backward compatibility
                 distributed: Optional[bool] = None, 
                 seed: Optional[int] = None,
                 **kwargs):
        
        # Auto-select strategy if not provided
        if resumption_strategy is None:
            if distributed is None:
                distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
            
            # Choose strategy based on context
            if distributed:
                resumption_strategy = DistributedResumptionStrategy()
            elif len(dataset) > 1_000_000:  # Large dataset
                resumption_strategy = AdvancedResumptionStrategy()
            else:
                resumption_strategy = SimpleResumptionStrategy()
        
        self.resumption_strategy = resumption_strategy
        
        # Create sampler if needed
        if sampler is None and batch_sampler is None:
            sampler = resumption_strategy.create_sampler(
                dataset,
                shuffle=shuffle if shuffle is not None else True,
                seed=seed
            )
            # Disable shuffle since we're using a custom sampler
            shuffle = False
        
        # Build DataLoader arguments
        dataloader_kwargs = {
            'dataset': dataset, 'batch_size': batch_size, 'shuffle': shuffle,
            'sampler': sampler, 'batch_sampler': batch_sampler, 'num_workers': num_workers,
            'collate_fn': collate_fn, 'pin_memory': pin_memory, 'drop_last': drop_last,
            'timeout': timeout, 'worker_init_fn': worker_init_fn,
            'multiprocessing_context': multiprocessing_context, 'generator': generator,
            'persistent_workers': persistent_workers
        }
        
        # Only add prefetch_factor if num_workers > 0
        if num_workers > 0:
            dataloader_kwargs['prefetch_factor'] = prefetch_factor
            
        super().__init__(**dataloader_kwargs, **kwargs)
        
        # Attach dataloader to strategy
        self.resumption_strategy.attach_dataloader(self)
        
        # Legacy attributes for backward compatibility
        self._batches_processed = 0  # Will be updated by strategy
        self._epoch = 0
        self._distributed = distributed
        
    def __iter__(self) -> Iterator:
        """Return an iterator that tracks progress using the strategy."""
        base_iter = super().__iter__()
        return self.resumption_strategy.wrap_iterator(base_iter)
    
    def state_dict(self) -> Dict[str, Any]:
        """Save DataLoader state using the strategy."""
        return self.resumption_strategy.state_dict()
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load DataLoader state using the strategy."""
        self.resumption_strategy.load_state_dict(state_dict)
        
        # Update legacy attributes for backward compatibility
        if isinstance(self.resumption_strategy, AdvancedResumptionStrategy):
            self._batches_processed = self.resumption_strategy._batches_processed
            self._epoch = self.resumption_strategy._epoch
    
    def set_epoch(self, epoch: int) -> None:
        """
        Set epoch for distributed training.
        
        Updates internal epoch tracking and forwards to sampler if supported.
        """
        self._epoch = epoch
        if hasattr(self.resumption_strategy, '_epoch'):
            self.resumption_strategy._epoch = epoch
        if hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)
    
    @property
    def batches_processed(self) -> int:
        """Get number of batches processed (for backward compatibility)."""
        if hasattr(self.resumption_strategy, '_batches_processed'):
            return self.resumption_strategy._batches_processed
        elif hasattr(self.resumption_strategy, 'batches_seen'):
            return self.resumption_strategy.batches_seen
        return self._batches_processed


def create_resumable_dataloader(dataset, batch_size: int, shuffle: bool = True,
                               num_workers: int = 0, pin_memory: bool = True,
                               drop_last: bool = False, 
                               strategy: Optional[str] = None,
                               distributed: Optional[bool] = None,
                               seed: Optional[int] = None, 
                               **kwargs) -> ResumableDataLoader:
    """
    Factory function to create a ResumableDataLoader with smart defaults.
    
    This function automatically selects the best strategy based on your
    environment and dataset characteristics.
    
    Args:
        dataset: Dataset to load from
        batch_size: Batch size for training
        shuffle: Whether to shuffle samples (default: True)
        num_workers: Number of worker processes (default: 0)
        pin_memory: Whether to pin memory for GPU transfer (default: True)
        drop_last: Whether to drop the last incomplete batch (default: False)
        strategy: Strategy name ('simple', 'advanced', 'distributed', None for auto)
        distributed: Whether using distributed training (default: auto-detect)
        seed: Random seed for reproducible shuffling (default: 42)
        **kwargs: Additional arguments passed to ResumableDataLoader
    
    Returns:
        ResumableDataLoader with appropriate strategy
        
    Example:
        >>> # Simple usage - auto-selects best strategy
        >>> dataloader = create_resumable_dataloader(
        ...     dataset=my_dataset,
        ...     batch_size=32
        ... )
        >>>
        >>> # Force simple strategy
        >>> dataloader = create_resumable_dataloader(
        ...     dataset=my_dataset,
        ...     batch_size=32,
        ...     strategy='simple'
        ... )
    """
    # Select strategy based on name or auto-detect
    if strategy == 'simple':
        resumption_strategy = SimpleResumptionStrategy()
    elif strategy == 'advanced':
        resumption_strategy = AdvancedResumptionStrategy()
    elif strategy == 'distributed':
        resumption_strategy = DistributedResumptionStrategy()
    else:
        # Auto-select strategy
        if distributed is None:
            distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        
        if distributed:
            resumption_strategy = DistributedResumptionStrategy()
        elif len(dataset) > 1_000_000:
            resumption_strategy = AdvancedResumptionStrategy()
        else:
            resumption_strategy = SimpleResumptionStrategy()
    
    return ResumableDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        resumption_strategy=resumption_strategy,
        seed=seed if seed is not None else 42,
        **kwargs
    )