"""
Additional test coverage for converter functionality edge cases and error handling.
"""

import pytest
import torch
from torch.utils.data import Dataset, IterableDataset
import numpy as np
from dataporter import ResumableDataLoader, create_resumable_dataloader, KeyBasedDtypeConverter
from dataporter.converters import KeyBasedDtypeConverter


class ListDataset(Dataset):
    """Dataset that returns lists instead of dicts."""
    def __init__(self, size=10):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return [
            torch.randn(3, 32, 32, dtype=torch.float32),  # Image
            torch.tensor(idx % 10, dtype=torch.int64),    # Label
            torch.ones(32, dtype=torch.float64)           # Weights
        ]


class TupleDataset(Dataset):
    """Dataset that returns tuples."""
    def __init__(self, size=10):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return (
            torch.randn(3, 32, 32, dtype=torch.float32),
            torch.tensor(idx % 10, dtype=torch.int64)
        )


class ScalarDataset(Dataset):
    """Dataset that returns single tensors."""
    def __init__(self, size=10):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return torch.randn(3, 224, 224, dtype=torch.float32)


class ComplexNestedDataset(Dataset):
    """Dataset with deeply nested structures."""
    def __init__(self, size=10):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            'level1': {
                'level2': {
                    'data': torch.randn(10, dtype=torch.float32),
                    'labels': torch.tensor([idx], dtype=torch.int64)
                },
                'metadata': {
                    'weights': torch.ones(5, dtype=torch.float64),
                    'flags': torch.tensor([1, 0, 1], dtype=torch.int64)
                }
            },
            'arrays': [
                torch.tensor([1.0, 2.0], dtype=torch.float32),
                torch.tensor([3, 4], dtype=torch.int64)
            ]
        }


class MixedTypeDataset(Dataset):
    """Dataset that returns mixed types including non-tensor data."""
    def __init__(self, size=10):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            'tensor': torch.randn(10, dtype=torch.float32),
            'string': f"sample_{idx}",
            'number': idx,
            'numpy': np.array([1, 2, 3]),
            'nested': {
                'tensor': torch.ones(5, dtype=torch.int64),
                'list': [1, 2, 3]
            }
        }


class TestConverterErrorHandling:
    """Test error handling and edge cases for converter functionality."""
    
    def test_none_converter(self):
        """Test that None converter is handled correctly."""
        dataset = ScalarDataset(10)
        
        # Should work with None converter
        dataloader = ResumableDataLoader(
            dataset,
            batch_size=2,
            converter=None
        )
        
        batch = next(iter(dataloader))
        assert batch.dtype == torch.float32
    
    def test_converter_with_list_dataset(self):
        """Test converter with dataset returning lists."""
        dataset = ListDataset(10)
        
        # Converter with list indices
        dataloader = ResumableDataLoader(
            dataset,
            batch_size=2,
            converter={
                '[0]': 'float16',  # First element (image)
                '[2]': 'float32'   # Third element (weights)
            }
        )
        
        batch = next(iter(dataloader))
        assert batch[0].dtype == torch.float16
        assert batch[1].dtype == torch.int64  # Unchanged
        assert batch[2].dtype == torch.float32
    
    def test_converter_with_tuple_dataset(self):
        """Test converter with dataset returning tuples."""
        dataset = TupleDataset(10)
        
        # Converter with tuple indices
        dataloader = ResumableDataLoader(
            dataset,
            batch_size=2,
            converter={
                '[0]': 'float16',  # Image
                '[1]': 'int32'     # Label
            }
        )
        
        batch = next(iter(dataloader))
        assert batch[0].dtype == torch.float16
        assert batch[1].dtype == torch.int32
    
    def test_converter_with_scalar_dataset(self):
        """Test converter with dataset returning single tensors."""
        dataset = ScalarDataset(10)
        
        # For scalar tensors, converter can't apply path-based conversion
        dataloader = ResumableDataLoader(
            dataset,
            batch_size=2,
            converter={'image': 'float16'}  # Won't match scalar tensor
        )
        
        batch = next(iter(dataloader))
        # Should remain unchanged since path doesn't match
        assert batch.dtype == torch.float32
    
    def test_converter_with_complex_nested_structure(self):
        """Test converter with deeply nested data structures."""
        dataset = ComplexNestedDataset(10)
        
        # Complex nested paths
        dataloader = ResumableDataLoader(
            dataset,
            batch_size=2,
            converter={
                'level1.level2.data': 'float16',
                'level1.level2.labels': 'int32',
                'level1.metadata.weights': 'float32',
                'level1.metadata.flags': 'uint8',
                'arrays[0]': 'float16',
                'arrays[1]': 'int16'
            }
        )
        
        batch = next(iter(dataloader))
        assert batch['level1']['level2']['data'].dtype == torch.float16
        assert batch['level1']['level2']['labels'].dtype == torch.int32
        assert batch['level1']['metadata']['weights'].dtype == torch.float32
        assert batch['level1']['metadata']['flags'].dtype == torch.uint8
        assert batch['arrays'][0].dtype == torch.float16
        assert batch['arrays'][1].dtype == torch.int16
    
    def test_converter_with_mixed_types(self):
        """Test converter handles non-tensor data gracefully."""
        dataset = MixedTypeDataset(10)
        
        dataloader = ResumableDataLoader(
            dataset,
            batch_size=2,
            converter={
                'tensor': 'float16',
                'string': 'float16',  # Can't convert string
                'nested.tensor': 'int32'
            }
        )
        
        batch = next(iter(dataloader))
        assert batch['tensor'].dtype == torch.float16
        # Check that strings are preserved (order may vary due to shuffling)
        assert all(isinstance(s, str) and s.startswith('sample_') for s in batch['string'])
        assert len(batch['string']) == 2
        # Check numbers are preserved - they get batched as tensors by default collate
        if isinstance(batch['number'], torch.Tensor):
            assert batch['number'].shape[0] == 2
        else:
            assert all(isinstance(n, int) for n in batch['number'])
            assert len(batch['number']) == 2
        assert isinstance(batch['numpy'], torch.Tensor)  # numpy arrays get converted to tensors
        assert batch['nested']['tensor'].dtype == torch.int32
    
    def test_converter_add_remove_rules(self):
        """Test dynamic converter rule management."""
        converter = KeyBasedDtypeConverter({'image': 'float16'})
        
        # Add new rule
        converter.add_conversion_rule('label', 'int32')
        assert 'label' in converter.dtype_map
        
        # Remove rule
        removed = converter.remove_conversion_rule('image')
        assert removed is True
        assert 'image' not in converter.dtype_map
        
        # Try to remove non-existent rule
        removed = converter.remove_conversion_rule('nonexistent')
        assert removed is False
    
    def test_converter_with_persistent_workers(self):
        """Test converter works with persistent workers."""
        dataset = MixedTypeDataset(20)
        
        dataloader = ResumableDataLoader(
            dataset,
            batch_size=4,
            num_workers=2,
            persistent_workers=True,
            converter={'tensor': 'float16', 'nested.tensor': 'int16'}
        )
        
        # Consume multiple epochs to test persistent workers
        for epoch in range(2):
            dataloader.set_epoch(epoch)
            batches = list(dataloader)
            
            # Verify conversions work across epochs
            for batch in batches:
                assert batch['tensor'].dtype == torch.float16
                assert batch['nested']['tensor'].dtype == torch.int16
    
    def test_converter_summary(self):
        """Test converter summary functionality."""
        converter = KeyBasedDtypeConverter({
            'image': 'float16',
            'label': 'int32',
            'mask': 'uint8'
        })
        
        summary = converter.get_conversion_summary()
        assert summary == {
            'image': 'float16',
            'label': 'int32',
            'mask': 'uint8'
        }
    
    def test_invalid_dtype_in_add_rule(self):
        """Test that adding invalid dtype raises error."""
        converter = KeyBasedDtypeConverter()
        
        with pytest.raises(ValueError, match="Unsupported dtype"):
            converter.add_conversion_rule('path', 'invalid_type')
    
    def test_converter_with_drop_last(self):
        """Test converter with drop_last=True."""
        dataset = ListDataset(15)
        
        dataloader = ResumableDataLoader(
            dataset,
            batch_size=4,
            drop_last=True,
            converter={'[0]': 'float16'}
        )
        
        batches = list(dataloader)
        # Should have 3 batches (15 // 4 = 3, last batch of 3 items dropped)
        assert len(batches) == 3
        
        # Verify conversions
        for batch in batches:
            assert batch[0].dtype == torch.float16


class TestConverterIntegrationScenarios:
    """Test real-world integration scenarios."""
    
    def test_converter_with_checkpoint_resume(self):
        """Test that converter works correctly across checkpoint/resume cycles."""
        dataset = ComplexNestedDataset(50)
        
        # Create first dataloader with converter
        dataloader1 = ResumableDataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            seed=42,
            converter={
                'level1.level2.data': 'float16',
                'level1.metadata.weights': 'float32'
            }
        )
        
        # Process some batches and collect results
        results1 = []
        iterator = iter(dataloader1)
        for _ in range(5):
            batch = next(iterator)
            results1.append({
                'data_dtype': batch['level1']['level2']['data'].dtype,
                'weights_dtype': batch['level1']['metadata']['weights'].dtype
            })
        
        # Save state
        state = dataloader1.state_dict()
        
        # Create new dataloader with same converter config
        dataloader2 = ResumableDataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            seed=42,
            converter={
                'level1.level2.data': 'float16',
                'level1.metadata.weights': 'float32'
            }
        )
        
        # Load state and continue
        dataloader2.load_state_dict(state)
        
        # Process more batches
        iterator2 = iter(dataloader2)
        for _ in range(5):
            batch = next(iterator2)
            # Verify conversions still work
            assert batch['level1']['level2']['data'].dtype == torch.float16
            assert batch['level1']['metadata']['weights'].dtype == torch.float32
    
    def test_converter_with_collate_fn(self):
        """Test converter works with custom collate functions."""
        dataset = ListDataset(20)
        
        def custom_collate(batch):
            # Custom collate that adds extra processing
            images = torch.stack([item[0] for item in batch])
            labels = torch.stack([item[1] for item in batch])
            weights = torch.stack([item[2] for item in batch])
            
            return {
                'images': images,
                'labels': labels,
                'weights': weights,
                'batch_size': len(batch)
            }
        
        dataloader = ResumableDataLoader(
            dataset,
            batch_size=4,
            collate_fn=custom_collate,
            converter={
                'images': 'float16',
                'labels': 'int32',
                'weights': 'float32'
            }
        )
        
        batch = next(iter(dataloader))
        assert batch['images'].dtype == torch.float16
        assert batch['labels'].dtype == torch.int32
        assert batch['weights'].dtype == torch.float32
        assert batch['batch_size'] == 4  # Non-tensor data preserved
    
    def test_empty_batches_with_converter(self):
        """Test converter handles empty datasets gracefully."""
        dataset = ListDataset(0)
        
        dataloader = ResumableDataLoader(
            dataset,
            batch_size=4,
            converter={'[0]': 'float16'}
        )
        
        batches = list(dataloader)
        assert len(batches) == 0  # No batches from empty dataset


if __name__ == "__main__":
    pytest.main([__file__, "-v"])