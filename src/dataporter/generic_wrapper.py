"""
Generic dataset wrapper with dtype conversion for any PyTorch dataset.
This provides a flexible way to apply dtype conversions to existing datasets.
"""

from torch.utils.data import Dataset
from typing import Optional, Dict, Any
from .base_wrapper import BaseDatasetWrapper
from .converters import KeyBasedDtypeConverter
import logging

logger = logging.getLogger(__name__)


class GenericDatasetWrapper(BaseDatasetWrapper):
    """
    A generic wrapper that applies dtype conversions to any PyTorch dataset.
    
    This is particularly useful for:
    - Reducing VRAM usage by converting to lower precision dtypes
    - Working with existing datasets without modifying their code
    - Applying conversions based on configurable paths
    
    Example:
        ```python
        # Wrap any existing dataset
        wrapped_dataset = GenericDatasetWrapper(
            original_dataset,
            dtype_conversions={
                "observation.image": "float16",  # 50% memory savings
                "action": "float16",
                "next.done": "uint8"  # Boolean only needs 1 byte
            }
        )
        ```
    """
    
    def _post_init(self, custom_path_mapping: Optional[Dict[str, str]] = None, **kwargs):
        """
        Additional initialization for GenericDatasetWrapper.
        
        Args:
            custom_path_mapping: Optional dict to remap paths before conversion
                               e.g. {"images": "observation.image"} to handle different naming
        """
        self.custom_path_mapping = custom_path_mapping or {}
    
    def __getitem__(self, idx: int) -> Any:
        """
        Get an item from the dataset and apply dtype conversions.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            The item with dtype conversions applied
        """
        # Get item from base dataset
        item = self.base_dataset[idx]
        
        # Process the item (applies path mapping if configured)
        processed_item = self._process_item(item)
        
        # Apply dtype conversions (handled by base class)
        return self._apply_dtype_conversions(processed_item)
    
    def _process_item(self, item: Any) -> Any:
        """
        Process the item, applying path mapping if configured.
        
        Args:
            item: Raw item from base dataset
            
        Returns:
            Processed item
        """
        # Apply custom path mapping if needed
        if self.custom_path_mapping:
            return self._apply_path_mapping(item)
        return item
    
    def _apply_path_mapping(self, item: Any, path: str = "") -> Any:
        """
        Apply custom path mapping to standardize paths before conversion.
        
        This allows handling datasets with different naming conventions.
        For example, mapping "images" -> "observation.image" for consistency.
        """
        if isinstance(item, dict):
            result = {}
            for key, value in item.items():
                current_path = f"{path}.{key}".lstrip('.')
                
                # Check if this path needs remapping
                if current_path in self.custom_path_mapping:
                    new_path = self.custom_path_mapping[current_path]
                    # Create nested structure for the new path
                    self._set_nested_value(result, new_path, value)
                else:
                    result[key] = self._apply_path_mapping(value, current_path)
            return result
        else:
            return item
    
    def _set_nested_value(self, target_dict: dict, path: str, value: Any) -> None:
        """
        Set a value in a nested dictionary using a dot-separated path.
        
        Args:
            target_dict: The dictionary to modify
            path: Dot-separated path (e.g., "observation.image")
            value: The value to set
        """
        keys = path.split('.')
        current = target_dict
        
        # Navigate/create nested structure
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the final value
        current[keys[-1]] = value


class DataLoaderDtypeWrapper:
    """
    A wrapper for DataLoader that ensures dtype conversions are applied.
    
    This is useful when you want to apply conversions at the DataLoader level
    rather than the dataset level, which can be more efficient for some use cases.
    """
    
    def __init__(self, dataloader, dtype_conversions: Optional[Dict[str, str]] = None):
        """
        Initialize the DataLoader wrapper.
        
        Args:
            dataloader: The underlying DataLoader
            dtype_conversions: Dict mapping paths to target dtypes
        """
        self.dataloader = dataloader
        self.dtype_converter = KeyBasedDtypeConverter(dtype_conversions)
    
    def __iter__(self):
        """Iterate through the dataloader, applying conversions to each batch."""
        for batch in self.dataloader:
            if self.dtype_converter.dtype_map:
                batch = self.dtype_converter.convert_batch(batch)
            yield batch
    
    def __len__(self):
        """Return the length of the underlying dataloader."""
        return len(self.dataloader)
    
    @property
    def batch_size(self):
        """Return the batch size of the underlying dataloader."""
        return self.dataloader.batch_size
    
    @property
    def dataset(self):
        """Return the dataset of the underlying dataloader."""
        return self.dataloader.dataset