"""
Key-based dtype conversion system for generic batch processing.
Supports any nested dictionary structure with configurable dtype mappings.
"""

import torch
from typing import Dict, Any, Union, Optional
import logging

logger = logging.getLogger(__name__)

class KeyBasedDtypeConverter:
    """Simple dtype converter using exact key paths."""
    
    def __init__(self, dtype_map: Optional[Dict[str, str]] = None):
        """
        Initialize the dtype converter.
        
        Args:
            dtype_map: Dict mapping key paths to target dtypes
                      e.g. {"inputs.image": "float16", "conditions.class_labels": "int64"}
                      If None, no conversions will be performed.
        """
        self.dtype_map = dtype_map or {}
        self.torch_dtypes = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16, 
            "float32": torch.float32,
            "float64": torch.float64,
            "int8": torch.int8,
            "int16": torch.int16,
            "int32": torch.int32,
            "int64": torch.int64,
            "uint8": torch.uint8,
            "uint16": torch.uint16,  # Added for token IDs optimization
            "bool": torch.bool,
        }
        
        # Validate dtype mappings
        self._validate_dtype_map()
        
        # Log configuration
        if self.dtype_map:
            logger.info(f"KeyBasedDtypeConverter initialized with {len(self.dtype_map)} conversion rules")
            for path, dtype in self.dtype_map.items():
                logger.debug(f"  {path} -> {dtype}")
        else:
            logger.info("KeyBasedDtypeConverter initialized with no conversion rules")
    
    def _validate_dtype_map(self):
        """Validate that all specified dtypes are supported."""
        for path, dtype_str in self.dtype_map.items():
            if dtype_str not in self.torch_dtypes:
                supported_dtypes = list(self.torch_dtypes.keys())
                raise ValueError(
                    f"Unsupported dtype '{dtype_str}' for path '{path}'. "
                    f"Supported dtypes: {supported_dtypes}"
                )
    
    def convert_batch(self, batch: Any) -> Any:
        """
        Convert batch based on key mappings.
        
        Args:
            batch: Input batch (typically a dictionary)
            
        Returns:
            Converted batch with same structure but updated dtypes
        """
        if not self.dtype_map:
            return batch
            
        return self._convert_recursive(batch, "")
    
    def _convert_recursive(self, obj: Any, current_path: str) -> Any:
        """
        Recursively convert based on current path.
        
        Args:
            obj: Current object to process
            current_path: Current path in the nested structure
            
        Returns:
            Converted object
        """
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                new_path = f"{current_path}.{key}".lstrip('.')
                result[key] = self._convert_recursive(value, new_path)
            return result
        
        elif isinstance(obj, (list, tuple)):
            # Handle lists/tuples by processing each element with indexed path
            result_items = []
            for i, item in enumerate(obj):
                indexed_path = f"{current_path}[{i}]"
                result_items.append(self._convert_recursive(item, indexed_path))
            return type(obj)(result_items)
        
        elif isinstance(obj, torch.Tensor):
            return self._convert_tensor(obj, current_path)
        
        else:
            # Return other types unchanged (strings, numbers, etc.)
            return obj
    
    def _convert_tensor(self, tensor: torch.Tensor, path: str) -> torch.Tensor:
        """
        Convert a single tensor if its path matches a conversion rule.
        
        Args:
            tensor: Input tensor
            path: Current path of the tensor
            
        Returns:
            Converted tensor or original tensor if no rule matches
        """
        # Check if current path matches any conversion rule
        target_dtype_str = self.dtype_map.get(path)
        if target_dtype_str is None:
            return tensor
        
        target_dtype = self.torch_dtypes[target_dtype_str]
        
        # Only convert if dtype actually differs
        if tensor.dtype == target_dtype:
            return tensor
        
        try:
            converted_tensor = tensor.to(target_dtype)
            logger.debug(f"Converted tensor at '{path}' from {tensor.dtype} to {target_dtype}")
            return converted_tensor
        except Exception as e:
            logger.warning(f"Failed to convert tensor at '{path}' from {tensor.dtype} to {target_dtype}: {e}")
            return tensor
    
    def get_conversion_summary(self) -> Dict[str, str]:
        """
        Get a summary of all configured conversions.
        
        Returns:
            Dictionary mapping paths to target dtypes
        """
        return self.dtype_map.copy()
    
    def add_conversion_rule(self, path: str, target_dtype: str):
        """
        Add a new conversion rule.
        
        Args:
            path: Key path to match
            target_dtype: Target dtype string
        """
        if target_dtype not in self.torch_dtypes:
            supported_dtypes = list(self.torch_dtypes.keys())
            raise ValueError(
                f"Unsupported dtype '{target_dtype}'. "
                f"Supported dtypes: {supported_dtypes}"
            )
        
        self.dtype_map[path] = target_dtype
        logger.info(f"Added conversion rule: {path} -> {target_dtype}")
    
    def remove_conversion_rule(self, path: str) -> bool:
        """
        Remove a conversion rule.
        
        Args:
            path: Key path to remove
            
        Returns:
            True if rule was removed, False if it didn't exist
        """
        if path in self.dtype_map:
            del self.dtype_map[path]
            logger.info(f"Removed conversion rule for path: {path}")
            return True
        return False