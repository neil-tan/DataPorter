"""
Unified dataset wrapper for HuggingFace datasets that handles both map-style and iterable datasets.
This replaces the complex multi-layer wrapping with a single, clean interface.
"""

from torch.utils.data import Dataset, IterableDataset
import sys
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
from .converters import KeyBasedDtypeConverter
import warnings
import logging

# Suppress DeepSpeed logging during worker initialization
logging.getLogger('deepspeed').setLevel(logging.ERROR)

# Lazy imports to reduce worker startup time
_torch = None
_transforms = None
_np = None
_Image = None

def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch

def _get_transforms():
    global _transforms
    if _transforms is None:
        from torchvision import transforms
        _transforms = transforms
    return _transforms

def _get_numpy():
    global _np
    if _np is None:
        import numpy as np
        _np = np
    return _np

def _get_image():
    global _Image
    if _Image is None:
        from PIL import Image
        _Image = Image
    return _Image


class UnifiedHFDatasetWrapper(Dataset):
    """
    A unified wrapper for HuggingFace datasets that handles:
    - Both map-style and iterable/streaming datasets
    - Image preprocessing and transformations
    - Condition processing (class labels, text captions, etc.)
    - Text tokenization
    - Proper epoch setting for shuffling
    """
    
    def __init__(
        self,
        hf_dataset,
        image_column_name: str,
        condition_column_name: Optional[str] = None,
        condition_type: str = "unconditional",  # "class_label", "text_caption", "text_from_label", "unconditional"
        image_transform: Optional[Callable] = None,
        tokenizer_name_or_path: Optional[str] = None,  # Pass config instead of object
        tokenizer_max_length: int = 512,
        class_names: Optional[List[str]] = None,
        caption_template: str = "A photo of a {}.",
        target_channels: int = 3,  # 1 for grayscale, 3 for RGB
        text_prompt_key: Optional[str] = None,
        dtype_conversions: Optional[Dict[str, str]] = None,  # NEW: dtype conversion rules
    ):
        """
        Initialize the unified dataset wrapper.
        
        Args:
            hf_dataset: The underlying HuggingFace dataset (map-style or iterable)
            image_column_name: Column name containing images
            condition_column_name: Column name containing conditions (labels/text)
            condition_type: Type of conditioning to apply
            image_transform: Optional image transformation pipeline
            tokenizer: Optional HuggingFace tokenizer for text processing
            tokenizer_max_length: Maximum length for tokenization
            class_names: List of class names for text_from_label conditioning
            caption_template: Template for converting labels to text
            target_channels: Number of color channels (1 or 3)
            text_prompt_key: Key to extract text from dict-type conditions
            dtype_conversions: Dict mapping batch paths to target dtypes (e.g. {"inputs.image": "float16"})
        """
        self.hf_dataset = hf_dataset
        self.image_column_name = image_column_name
        self.condition_column_name = condition_column_name
        self.condition_type = condition_type
        self.image_transform = image_transform
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.tokenizer_max_length = tokenizer_max_length
        self._tokenizer = None  # Lazy-loaded tokenizer
        self.class_names = class_names
        self.caption_template = caption_template
        self.target_channels = target_channels
        self.text_prompt_key = text_prompt_key
        
        # Initialize dtype converter
        self.dtype_converter = KeyBasedDtypeConverter(dtype_conversions)
        
        # Detect if this is an iterable dataset
        self.is_iterable = isinstance(hf_dataset, IterableDataset) or \
                          hasattr(hf_dataset, '_ex_iterable') or \
                          not hasattr(hf_dataset, '__len__')
        
        # Validate dataset structure if possible
        self._validate_dataset_structure()
        
        # Create default transform if none provided (lazy)
        if self.image_transform is None:
            self.image_transform = self._create_default_transform()
    
    @property
    def tokenizer(self):
        """Lazy-load tokenizer to avoid pickling issues in multiprocessing."""
        if self._tokenizer is None and self.tokenizer_name_or_path:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path, use_fast=True)
            except ImportError:
                print("[WARNING] transformers library not found. Text tokenization will not be available.")
                self._tokenizer = None
            except Exception as e:
                print(f"[WARNING] Could not load tokenizer '{self.tokenizer_name_or_path}': {e}")
                self._tokenizer = None
        return self._tokenizer
    
    def _validate_dataset_structure(self):
        """Validate that the dataset has the expected columns."""
        # Skip validation for iterable datasets or those without features
        if self.is_iterable or not hasattr(self.hf_dataset, 'features'):
            return
            
        features = self.hf_dataset.features
        if features is None:
            return
            
        # Check image column
        if self.image_column_name not in features:
            raise ValueError(f"Image column '{self.image_column_name}' not found in dataset")
        
        # Check condition column if needed
        if self.condition_type != "unconditional" and self.condition_column_name:
            if self.condition_column_name not in features:
                raise ValueError(f"Condition column '{self.condition_column_name}' not found in dataset")
        
        # Auto-infer class names for text_from_label if possible
        if self.condition_type == "text_from_label" and not self.class_names:
            if hasattr(features.get(self.condition_column_name), 'names'):
                self.class_names = features[self.condition_column_name].names
                print(f"[INFO] Auto-inferred class names: {self.class_names}")
    
    def _create_default_transform(self):
        """Create a default image transformation pipeline."""
        transforms = _get_transforms()
        transform_list = []
        
        # Handle grayscale conversion
        if self.target_channels == 1:
            transform_list.append(transforms.Grayscale(num_output_channels=1))
            normalize = transforms.Normalize((0.5,), (0.5,))
        else:
            # Ensure 3 channels even for grayscale inputs
            transform_list.append(transforms.Lambda(lambda x: x.convert("RGB") if hasattr(x, 'convert') else x))
            normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
        transform_list.extend([transforms.ToTensor(), normalize])
        return transforms.Compose(transform_list)
    
    def set_epoch(self, epoch: int):
        """
        Set the epoch for proper shuffling in distributed/streaming settings.
        This method propagates the epoch to the underlying dataset.
        """
        if hasattr(self.hf_dataset, 'set_epoch'):
            self.hf_dataset.set_epoch(epoch)
            print(f"[INFO] Set epoch {epoch} for dataset shuffling")
    
    def _process_image(self, image_data: Any):
        """Process raw image data into a tensor using optimized operations."""
        torch = _get_torch()
        np = _get_numpy()
        Image = _get_image()
        
        # Convert numpy arrays to PIL (vectorized when possible)
        if isinstance(image_data, np.ndarray):
            # For batch processing, use optimized numpy operations
            if image_data.ndim == 4:  # Batch of images
                processed_batch = []
                for img in image_data:
                    processed_batch.append(self.image_transform(Image.fromarray(img)))
                return torch.stack(processed_batch)
            else:
                image_data = Image.fromarray(image_data)
        
        # Ensure correct color mode for PIL images
        if hasattr(image_data, 'convert'):
            if self.target_channels == 3 and image_data.mode != "RGB":
                image_data = image_data.convert("RGB")
            elif self.target_channels == 1 and image_data.mode != "L":
                image_data = image_data.convert("L")
        
        # Apply transforms
        return self.image_transform(image_data)
    
    def _process_condition(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process condition data based on condition type with optimized operations."""
        condition_dict = {}
        
        if self.condition_type == "unconditional" or not self.condition_column_name:
            return condition_dict
        
        if self.condition_type == "class_label":
            label = item[self.condition_column_name]
            # Use optimized tensor creation for better performance
            torch = _get_torch()
            condition_dict['class_labels'] = torch.tensor(label, dtype=torch.long)
            
        elif self.condition_type == "text_caption":
            text = self._extract_text(item[self.condition_column_name])
            condition_dict['text'] = text
            
        elif self.condition_type == "text_from_label":
            label = item[self.condition_column_name]
            class_name = "unknown"
            if self.class_names and 0 <= label < len(self.class_names):
                class_name = self.class_names[label]
            text = self.caption_template.format(class_name)
            condition_dict['text'] = text
            
        else:
            raise ValueError(f"Unsupported condition_type: {self.condition_type}")
        
        # Apply tokenization with optimized settings for multiprocessing
        if 'text' in condition_dict and self.tokenizer:
            # Use fast tokenizer with optimized settings
            tokenized = self.tokenizer(
                condition_dict['text'],
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer_max_length,
                return_tensors="pt",
                return_attention_mask=True,
                # Enable fast tokenizer optimizations
                add_special_tokens=True,
                return_token_type_ids=False  # Reduces memory usage
            )
            # Remove batch dimension efficiently
            condition_dict['tokenized'] = {k: v.squeeze(0) for k, v in tokenized.items()}
        
        return condition_dict
    
    def _extract_text(self, value: Any) -> str:
        """Extract text from various data formats."""
        if self.text_prompt_key and isinstance(value, dict):
            return value.get(self.text_prompt_key, "")
        elif isinstance(value, list):
            return value[0] if value else ""
        elif isinstance(value, str):
            return value
        else:
            return str(value)
    
    def _process_item(self, item) -> Dict[str, Any]:
        """Process a single item from the dataset."""
        # Handle different item formats
        if isinstance(item, dict):
            # Standard dictionary format from HuggingFace datasets
            item_dict = item
        elif isinstance(item, tuple):
            # Legacy tuple format - convert to dict
            if len(item) >= 2:
                item_dict = {
                    self.image_column_name: item[0],
                    self.condition_column_name: item[1] if len(item) > 1 else {}
                }
            else:
                raise ValueError(f"Unexpected tuple format: {item}")
        else:
            raise ValueError(f"Unsupported item format: {type(item)}")
        
        # Process image
        image_tensor = self._process_image(item_dict[self.image_column_name])
        
        # Process conditions
        condition_dict = self._process_condition(item_dict)
        
        # Return in new dictionary format
        batch = {
            'inputs': {'image': image_tensor},
            'conditions': condition_dict
        }
        
        # Apply dtype conversions
        return self.dtype_converter.convert_batch(batch)
    
    def __len__(self):
        """Return dataset length. For iterable datasets, return a large number."""
        if self.is_iterable:
            return 2**31 - 1  # Large number for iterable datasets
        return len(self.hf_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single item by index (map-style datasets only)."""
        if self.is_iterable:
            raise TypeError("Cannot index an IterableDataset. Use iteration instead.")
        
        item = self.hf_dataset[idx]
        return self._process_item(item)
    
    def __iter__(self):
        """Iterate through the dataset (works for both map-style and iterable)."""
        if self.is_iterable:
            # For iterable datasets, iterate through the underlying dataset
            for item in self.hf_dataset:
                yield self._process_item(item)
        else:
            # For map-style datasets, provide iteration support
            for idx in range(len(self)):
                yield self[idx]