# DataPorter 🚀

PyTorch data loading utilities for seamless training resumption and memory optimization.

[![PyPI version](https://badge.fury.io/py/dataporter.svg)](https://badge.fury.io/py/dataporter)
[![Python Version](https://img.shields.io/pypi/pyversions/dataporter.svg)](https://pypi.org/project/dataporter/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Why DataPorter?

Training large models is expensive and time-consuming. When training gets interrupted (preemption, errors, or manual pauses), you shouldn't have to start over. DataPorter ensures you can resume training from the exact same data sample, maintaining reproducibility and saving valuable time and compute resources.

### Key Features

- **🔄 Exact Resume**: Resume training from the exact sample where you left off
- **💾 Memory Efficient**: Reduce memory usage by 50-87% with smart dtype conversions
- **🎯 Strategy Pattern**: Choose between simple, advanced, or distributed resumption strategies
- **🔧 Drop-in Replacement**: Works as a direct replacement for PyTorch's DataLoader
- **⚡ Production Ready**: Battle-tested in large-scale training environments

## Installation

```bash
pip install dataporter
```

For development:
```bash
pip install dataporter[dev]
```

For examples with HuggingFace integration:
```bash
pip install dataporter[examples]
```

## Quick Start

### Basic Usage

```python
from dataporter import create_resumable_dataloader

# Create a resumable dataloader - it's that simple!
dataloader = create_resumable_dataloader(
    dataset, 
    batch_size=32,
    shuffle=True
)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Your training code here
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
        
        # Save checkpoint with dataloader state
        if step % save_interval == 0:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'dataloader': dataloader.state_dict(),  # Save exact position
            }, 'checkpoint.pt')
```

### Resume Training

```python
# Load checkpoint
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
dataloader.load_state_dict(checkpoint['dataloader'])  # Resume from exact position

# Continue training from where you left off
for batch in dataloader:
    # Continues from the exact sample where it stopped
    train_step(batch)
```

## Advanced Features

### 1. Memory Optimization with Dtype Conversion

Reduce memory usage by up to 87% with intelligent dtype conversions:

```python
from dataporter import KeyBasedDtypeConverter

# Define conversion rules
converter = KeyBasedDtypeConverter({
    "inputs.image": "float16",           # Images to half precision (50% reduction)
    "labels": "int32",                    # Labels from int64 to int32 (50% reduction)
    "attention_mask": "uint8",            # Masks to uint8 (87.5% reduction!)
})

# Apply conversions
batch = converter.convert_batch(batch)
```

### 2. Different Resumption Strategies

Choose the strategy that fits your needs:

```python
# Simple strategy - Low memory overhead, batch-level precision
dataloader = create_resumable_dataloader(dataset, strategy='simple')

# Advanced strategy - Sample-level precision, distributed training support
dataloader = create_resumable_dataloader(dataset, strategy='advanced')

# Distributed strategy - Multi-GPU training with exact resume
dataloader = create_resumable_dataloader(dataset, strategy='distributed')
```

### 3. Integration with HuggingFace Datasets

```python
from datasets import load_dataset
from dataporter import UnifiedHFDatasetWrapper, create_resumable_dataloader

# Load HuggingFace dataset
hf_dataset = load_dataset("fashion_mnist", split="train")

# Wrap with dtype conversions
dataset = UnifiedHFDatasetWrapper(
    hf_dataset,
    dtype_conversions={
        "image": "float16",
        "label": "int32"
    }
)

# Create resumable dataloader
dataloader = create_resumable_dataloader(dataset, batch_size=64)
```

## Memory Savings Examples

| Data Type | Original | Converted | Memory Saved |
|-----------|----------|-----------|--------------|
| Images | float32 | float16 | 50% |
| Labels | int64 | int32 | 50% |
| Attention Masks | int64 | uint8 | 87.5% |
| Token IDs | int64 | int32 | 50% |

## Architecture

DataPorter uses a clean strategy pattern to separate concerns:

```
ResumableDataLoader
    ├── SimpleResumptionStrategy      # Batch counting, low overhead
    ├── AdvancedResumptionStrategy    # Sample-level precision
    └── DistributedResumptionStrategy # Multi-GPU support
```

Each strategy handles:
- State persistence and restoration
- Iterator wrapping for resume capability
- Distributed synchronization (if applicable)

## Examples

Check out the `examples/` directory for:
- [Basic resumable training](examples/basic_resume.py)
- [Distributed training with exact resume](examples/distributed_training.py)
- [Memory optimization techniques](examples/memory_optimization.py)
- [HuggingFace integration](examples/huggingface_integration.py)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

DataPorter is released under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use DataPorter in your research, please cite:

```bibtex
@software{dataporter2024,
  title = {DataPorter: PyTorch Data Loading Utilities for Seamless Training},
  author = {Tan, Neil},
  year = {2024},
  url = {https://github.com/yourusername/dataporter}
}
```

## Acknowledgments

DataPorter was born from the Yggdrasil project, where the need for robust training resumption became critical for long-running experiments.