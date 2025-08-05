"""
Test suite for converter integration with ResumableDataLoader.

Tests cover:
- Direct converter instance usage
- Dictionary-based converter creation
- Converter application in data loading pipeline
- State persistence with converters
"""

import pytest
import torch
from torch.utils.data import Dataset
from dataporter import ResumableDataLoader, create_resumable_dataloader, KeyBasedDtypeConverter


class DummyDatasetWithTypes(Dataset):
    """Dataset that returns various data types for testing conversions."""
    
    def __init__(self, size=100):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            'image': torch.randn(3, 224, 224, dtype=torch.float32),  # Simulate image
            'label': torch.tensor(idx % 10, dtype=torch.int64),      # Class label
            'mask': torch.ones(224, 224, dtype=torch.int64),         # Binary mask
            'metadata': {
                'idx': torch.tensor(idx, dtype=torch.int64),
                'weight': torch.tensor(1.0, dtype=torch.float64)
            }
        }


class TestConverterIntegration:
    """Test converter integration with ResumableDataLoader."""
    
    def test_converter_with_dict_initialization(self):
        """Test that converter can be initialized from a dict."""
        dataset = DummyDatasetWithTypes(10)
        
        # Create dataloader with dict-based converter
        dataloader = ResumableDataLoader(
            dataset,
            batch_size=2,
            converter={
                'image': 'float16',
                'label': 'int32',
                'mask': 'uint8'
            }
        )
        
        # Get a batch and verify conversions
        batch = next(iter(dataloader))
        
        assert batch['image'].dtype == torch.float16
        assert batch['label'].dtype == torch.int32
        assert batch['mask'].dtype == torch.uint8
        # Metadata should remain unchanged
        assert batch['metadata']['idx'].dtype == torch.int64
        assert batch['metadata']['weight'].dtype == torch.float64
    
    def test_converter_with_instance(self):
        """Test that converter instance can be passed directly."""
        dataset = DummyDatasetWithTypes(10)
        
        # Create converter instance
        converter = KeyBasedDtypeConverter({
            'image': 'bfloat16',
            'metadata.weight': 'float32'
        })
        
        # Create dataloader with converter instance
        dataloader = ResumableDataLoader(
            dataset,
            batch_size=2,
            converter=converter
        )
        
        # Get a batch and verify conversions
        batch = next(iter(dataloader))
        
        assert batch['image'].dtype == torch.bfloat16
        assert batch['metadata']['weight'].dtype == torch.float32
        # Others should remain unchanged
        assert batch['label'].dtype == torch.int64
        assert batch['mask'].dtype == torch.int64
    
    def test_create_function_with_converter(self):
        """Test that create_resumable_dataloader supports converter."""
        dataset = DummyDatasetWithTypes(10)
        
        # Use factory function with converter
        dataloader = create_resumable_dataloader(
            dataset,
            batch_size=2,
            converter={
                'image': 'float16',
                'label': 'int16',
                'mask': 'bool'
            }
        )
        
        # Get a batch and verify conversions
        batch = next(iter(dataloader))
        
        assert batch['image'].dtype == torch.float16
        assert batch['label'].dtype == torch.int16
        assert batch['mask'].dtype == torch.bool
    
    def test_no_converter_preserves_dtypes(self):
        """Test that without converter, original dtypes are preserved."""
        dataset = DummyDatasetWithTypes(10)
        
        # Create dataloader without converter
        dataloader = ResumableDataLoader(dataset, batch_size=2)
        
        # Get a batch and verify original dtypes
        batch = next(iter(dataloader))
        
        assert batch['image'].dtype == torch.float32
        assert batch['label'].dtype == torch.int64
        assert batch['mask'].dtype == torch.int64
        assert batch['metadata']['idx'].dtype == torch.int64
        assert batch['metadata']['weight'].dtype == torch.float64
    
    def test_converter_with_resumption(self):
        """Test that converter works correctly after resumption."""
        dataset = DummyDatasetWithTypes(20)
        
        # Create first dataloader with converter
        dataloader1 = ResumableDataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            seed=42,
            converter={'image': 'float16', 'label': 'int32'}
        )
        
        # Consume some batches
        iterator = iter(dataloader1)
        for _ in range(3):
            batch = next(iterator)
            assert batch['image'].dtype == torch.float16
            assert batch['label'].dtype == torch.int32
        
        # Save state
        state = dataloader1.state_dict()
        
        # Create new dataloader with same converter
        dataloader2 = ResumableDataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            seed=42,
            converter={'image': 'float16', 'label': 'int32'}
        )
        
        # Load state
        dataloader2.load_state_dict(state)
        
        # Continue iteration and verify conversions still work
        batch = next(iter(dataloader2))
        assert batch['image'].dtype == torch.float16
        assert batch['label'].dtype == torch.int32
    
    def test_converter_memory_savings(self):
        """Test that converter actually reduces memory usage."""
        dataset = DummyDatasetWithTypes(10)
        
        # Get original batch size
        dataloader_original = ResumableDataLoader(dataset, batch_size=1)
        batch_original = next(iter(dataloader_original))
        
        # Calculate original memory
        original_image_bytes = batch_original['image'].element_size() * batch_original['image'].nelement()
        original_label_bytes = batch_original['label'].element_size() * batch_original['label'].nelement()
        original_mask_bytes = batch_original['mask'].element_size() * batch_original['mask'].nelement()
        
        # Get converted batch
        dataloader_converted = ResumableDataLoader(
            dataset,
            batch_size=1,
            converter={
                'image': 'float16',  # 32->16 bits (50% reduction)
                'label': 'int32',    # 64->32 bits (50% reduction)
                'mask': 'uint8'      # 64->8 bits (87.5% reduction)
            }
        )
        batch_converted = next(iter(dataloader_converted))
        
        # Calculate converted memory
        converted_image_bytes = batch_converted['image'].element_size() * batch_converted['image'].nelement()
        converted_label_bytes = batch_converted['label'].element_size() * batch_converted['label'].nelement()
        converted_mask_bytes = batch_converted['mask'].element_size() * batch_converted['mask'].nelement()
        
        # Verify memory reductions
        assert converted_image_bytes == original_image_bytes // 2  # 50% reduction
        assert converted_label_bytes == original_label_bytes // 2  # 50% reduction
        assert converted_mask_bytes == original_mask_bytes // 8    # 87.5% reduction
    
    def test_invalid_dtype_raises_error(self):
        """Test that invalid dtype in converter raises appropriate error."""
        dataset = DummyDatasetWithTypes(10)
        
        # Should raise ValueError for invalid dtype
        with pytest.raises(ValueError, match="Unsupported dtype"):
            ResumableDataLoader(
                dataset,
                batch_size=2,
                converter={'image': 'invalid_dtype'}
            )
    
    def test_converter_with_distributed(self):
        """Test that converter works with distributed environment detection."""
        dataset = DummyDatasetWithTypes(10)
        
        # Create dataloader with converter (will auto-detect non-distributed)
        dataloader = ResumableDataLoader(
            dataset,
            batch_size=2,
            converter={'image': 'float16'}
        )
        
        # Should work normally
        batch = next(iter(dataloader))
        assert batch['image'].dtype == torch.float16


class TestConverterEdgeCases:
    """Test edge cases for converter functionality."""
    
    def test_empty_converter_dict(self):
        """Test that empty converter dict behaves like no converter."""
        dataset = DummyDatasetWithTypes(10)
        
        dataloader = ResumableDataLoader(
            dataset,
            batch_size=2,
            converter={}
        )
        
        # Should preserve original dtypes
        batch = next(iter(dataloader))
        assert batch['image'].dtype == torch.float32
        assert batch['label'].dtype == torch.int64
    
    def test_converter_with_nonexistent_paths(self):
        """Test that converter handles non-existent paths gracefully."""
        dataset = DummyDatasetWithTypes(10)
        
        dataloader = ResumableDataLoader(
            dataset,
            batch_size=2,
            converter={
                'image': 'float16',          # Exists
                'nonexistent.path': 'int32'  # Does not exist
            }
        )
        
        # Should apply conversions for existing paths only
        batch = next(iter(dataloader))
        assert batch['image'].dtype == torch.float16
        assert batch['label'].dtype == torch.int64  # Unchanged
    
    def test_converter_with_multiple_workers(self):
        """Test that converter works correctly with multiple workers."""
        dataset = DummyDatasetWithTypes(10)
        
        dataloader = ResumableDataLoader(
            dataset,
            batch_size=2,
            num_workers=2,
            converter={'image': 'float16', 'label': 'int32'}
        )
        
        # Consume all batches
        all_batches = list(dataloader)
        
        # Verify all batches have correct conversions
        for batch in all_batches:
            assert batch['image'].dtype == torch.float16
            assert batch['label'].dtype == torch.int32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])