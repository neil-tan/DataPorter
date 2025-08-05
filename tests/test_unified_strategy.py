"""
Comprehensive test suite for the unified resumption strategy.

Tests cover:
- Automatic detection of distributed vs single-node environments
- State persistence and restoration
- Epoch overflow handling
- Backward compatibility
- Integration with ResumableDataLoader
"""

import pytest
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from unittest.mock import Mock, patch, MagicMock
import warnings

from dataporter.strategies import UnifiedResumptionStrategy
from dataporter.samplers import ResumableSampler, ResumableDistributedSampler
from dataporter.resumable import ResumableDataLoader, create_resumable_dataloader


class DummyDataset(Dataset):
    """Simple dataset for testing."""
    def __init__(self, size=100):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return torch.tensor(idx)


class TestUnifiedResumptionStrategy:
    """Test the unified resumption strategy."""
    
    def test_auto_detects_single_node_environment(self):
        """Test that strategy correctly detects single-node environment."""
        with patch('torch.distributed.is_available', return_value=False):
            strategy = UnifiedResumptionStrategy()
            assert not strategy._is_distributed
    
    def test_auto_detects_distributed_environment(self):
        """Test that strategy correctly detects distributed environment."""
        with patch('torch.distributed.is_available', return_value=True), \
             patch('torch.distributed.is_initialized', return_value=True):
            strategy = UnifiedResumptionStrategy()
            assert strategy._is_distributed
    
    def test_creates_correct_sampler_single_node(self):
        """Test that correct sampler is created for single-node."""
        with patch('torch.distributed.is_available', return_value=False):
            strategy = UnifiedResumptionStrategy()
            dataset = DummyDataset(100)
            
            sampler = strategy.create_sampler(dataset, shuffle=True, seed=42)
            
            assert isinstance(sampler, ResumableSampler)
            assert not isinstance(sampler, ResumableDistributedSampler)
    
    def test_creates_correct_sampler_distributed(self):
        """Test that correct sampler is created for distributed."""
        with patch('torch.distributed.is_available', return_value=True), \
             patch('torch.distributed.is_initialized', return_value=True), \
             patch('torch.distributed.get_world_size', return_value=1), \
             patch('torch.distributed.get_rank', return_value=0):
            strategy = UnifiedResumptionStrategy()
            dataset = DummyDataset(100)
            
            sampler = strategy.create_sampler(dataset, shuffle=True, seed=42)
            
            assert isinstance(sampler, ResumableDistributedSampler)
    
    def test_state_dict_saves_all_required_state(self):
        """Test that state_dict saves all necessary information."""
        strategy = UnifiedResumptionStrategy()
        strategy._batches_processed = 50
        strategy._epoch = 3
        
        # Mock dataloader with sampler
        mock_sampler = Mock()
        mock_sampler.state_dict.return_value = {'sampler_state': 'test'}
        mock_dataloader = Mock()
        mock_dataloader.sampler = mock_sampler
        strategy.dataloader = mock_dataloader
        
        state = strategy.state_dict()
        
        assert state['batches_processed'] == 50
        assert state['epoch'] == 3
        assert 'distributed' in state
        assert state['sampler_state'] == {'sampler_state': 'test'}
    
    def test_load_state_dict_restores_state(self):
        """Test that load_state_dict properly restores state."""
        strategy = UnifiedResumptionStrategy()
        
        # Mock dataloader
        mock_dataloader = Mock()
        mock_dataloader.batch_size = 32
        mock_dataloader.dataset = DummyDataset(100)
        mock_sampler = Mock(spec=ResumableSampler)
        mock_sampler.start_sample = 0
        mock_sampler.current_epoch = 0
        mock_dataloader.sampler = mock_sampler
        strategy.dataloader = mock_dataloader
        
        # Load state - use values that don't trigger epoch overflow
        # 2 batches * 32 = 64 samples < 100 dataset size
        state = {
            'batches_processed': 2,
            'epoch': 1,
            'distributed': False
        }
        strategy.load_state_dict(state)
        
        assert strategy._batches_processed == 2
        assert strategy._epoch == 1
    
    def test_epoch_overflow_handling(self):
        """Test that epoch overflow is handled correctly."""
        strategy = UnifiedResumptionStrategy()
        
        # Mock dataloader - dataset size 100, batch size 32
        mock_dataloader = Mock()
        mock_dataloader.batch_size = 32
        mock_dataloader.dataset = DummyDataset(100)
        mock_sampler = Mock(spec=ResumableSampler)
        mock_sampler.start_sample = 0
        mock_sampler.current_epoch = 0
        mock_dataloader.sampler = mock_sampler
        strategy.dataloader = mock_dataloader
        
        # Load state where we've processed more samples than dataset size
        # 150 batches * 32 batch_size = 4800 samples
        # With dataset size 100, this is 48 complete epochs
        state = {
            'batches_processed': 150,
            'epoch': 0,
            'distributed': False
        }
        
        strategy.load_state_dict(state)
        
        # Should advance to epoch 48
        assert strategy._epoch == 48
        # Should be at batch 0 of the new epoch (4800 % 100 = 0)
        assert strategy._batches_processed == 0
    
    def test_distributed_mismatch_warning(self):
        """Test that warning is issued when distributed state mismatches."""
        # Create strategy in single-node environment
        with patch('torch.distributed.is_available', return_value=False):
            strategy = UnifiedResumptionStrategy()
        
        # Try to load distributed checkpoint
        state = {
            'batches_processed': 10,
            'epoch': 1,
            'distributed': True  # Mismatch!
        }
        
        with pytest.warns(RuntimeWarning, match="Distributed state mismatch"):
            strategy.load_state_dict(state)
    
    def test_iterator_wrapper_increments_batches(self):
        """Test that the iterator wrapper correctly tracks batches."""
        strategy = UnifiedResumptionStrategy()
        
        # Create mock iterator
        mock_data = [torch.tensor(i) for i in range(5)]
        mock_iterator = iter(mock_data)
        
        # Wrap iterator
        wrapped = strategy.wrap_iterator(mock_iterator)
        
        # Consume all batches
        batches = list(wrapped)
        
        assert len(batches) == 5
        assert strategy._batches_processed == 5
    
    def test_set_epoch_updates_sampler(self):
        """Test that set_epoch updates both strategy and sampler."""
        strategy = UnifiedResumptionStrategy()
        
        # Mock dataloader with sampler
        mock_sampler = Mock()
        mock_dataloader = Mock()
        mock_dataloader.sampler = mock_sampler
        strategy.dataloader = mock_dataloader
        
        strategy.set_epoch(5)
        
        assert strategy._epoch == 5
        mock_sampler.set_epoch.assert_called_once_with(5)


class TestBackwardCompatibility:
    """Test backward compatibility with old strategy names."""
    
    def test_simple_strategy_alias(self):
        """Test that SimpleResumptionStrategy is aliased to UnifiedResumptionStrategy."""
        from dataporter.strategies import SimpleResumptionStrategy
        
        strategy = SimpleResumptionStrategy()
        assert isinstance(strategy, UnifiedResumptionStrategy)
    
    def test_advanced_strategy_alias(self):
        """Test that AdvancedResumptionStrategy is aliased to UnifiedResumptionStrategy."""
        from dataporter.strategies import AdvancedResumptionStrategy
        
        strategy = AdvancedResumptionStrategy()
        assert isinstance(strategy, UnifiedResumptionStrategy)
    
    def test_distributed_strategy_alias(self):
        """Test that DistributedResumptionStrategy is aliased to UnifiedResumptionStrategy."""
        from dataporter.strategies import DistributedResumptionStrategy
        
        strategy = DistributedResumptionStrategy()
        assert isinstance(strategy, UnifiedResumptionStrategy)


class TestResumableDataLoaderIntegration:
    """Test integration with ResumableDataLoader."""
    
    def test_dataloader_uses_unified_strategy_by_default(self):
        """Test that ResumableDataLoader uses UnifiedResumptionStrategy by default."""
        dataset = DummyDataset(100)
        dataloader = ResumableDataLoader(dataset, batch_size=32)
        
        assert isinstance(dataloader.resumption_strategy, UnifiedResumptionStrategy)
    
    def test_create_function_shows_deprecation_warning(self):
        """Test that create_resumable_dataloader shows deprecation warning for strategy parameter."""
        dataset = DummyDataset(100)
        
        with pytest.warns(DeprecationWarning, match="The 'strategy' parameter is deprecated"):
            dataloader = create_resumable_dataloader(
                dataset, 
                batch_size=32,
                strategy='simple'  # This should trigger warning
            )
        
        # Should still work and use UnifiedResumptionStrategy
        assert isinstance(dataloader.resumption_strategy, UnifiedResumptionStrategy)
    
    def test_create_function_no_warning_without_strategy(self):
        """Test that no warning is shown when strategy parameter is not used."""
        dataset = DummyDataset(100)
        
        # Should not produce any warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            dataloader = create_resumable_dataloader(
                dataset,
                batch_size=32
            )
        
        assert isinstance(dataloader.resumption_strategy, UnifiedResumptionStrategy)
    
    def test_full_save_load_cycle(self):
        """Test complete save/load cycle with ResumableDataLoader."""
        dataset = DummyDataset(100)
        dataloader1 = ResumableDataLoader(dataset, batch_size=32, shuffle=True, seed=42)
        
        # Consume some batches
        iterator = iter(dataloader1)
        for _ in range(2):
            next(iterator)
        
        # Save state
        state = dataloader1.state_dict()
        
        # Create new dataloader and load state
        dataloader2 = ResumableDataLoader(dataset, batch_size=32, shuffle=True, seed=42)
        dataloader2.load_state_dict(state)
        
        # Verify state was restored
        assert dataloader2._batches_processed == 2
        assert dataloader2.resumption_strategy._batches_processed == 2


class TestDistributedScenarios:
    """Test distributed training scenarios."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_distributed_detection_with_cuda(self):
        """Test distributed detection when CUDA is available."""
        # This test would require actual distributed setup
        # For now, we just test the detection logic
        with patch('torch.distributed.is_available', return_value=True), \
             patch('torch.distributed.is_initialized', return_value=False):
            strategy = UnifiedResumptionStrategy()
            assert not strategy._is_distributed  # Not initialized = not distributed
    
    def test_sampler_state_update_distributed(self):
        """Test that distributed sampler state is updated correctly."""
        with patch('torch.distributed.is_available', return_value=True), \
             patch('torch.distributed.is_initialized', return_value=True):
            strategy = UnifiedResumptionStrategy()
            
            # Mock distributed sampler
            mock_sampler = Mock(spec=ResumableDistributedSampler)
            mock_dataloader = Mock()
            mock_dataloader.sampler = mock_sampler
            mock_dataloader.batch_size = 32
            mock_dataloader.dataset = DummyDataset(100)
            strategy.dataloader = mock_dataloader
            
            # Update sampler state
            strategy._update_sampler_state(50, 2, {})
            
            # Verify all required attributes were set
            assert mock_sampler.start_sample == 50
            assert mock_sampler.current_epoch == 2
            assert mock_sampler.start_epoch == 2
            mock_sampler.set_epoch.assert_called_once_with(2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])