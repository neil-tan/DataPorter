"""
Basic example of using DataPorter for resumable training
"""

import torch
from torch.utils.data import Dataset
from dataporter import create_resumable_dataloader


class SimpleDataset(Dataset):
    """A simple dataset for demonstration"""
    def __init__(self, size=1000):
        self.size = size
        self.data = torch.randn(size, 10)
        self.labels = torch.randint(0, 2, (size,))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def main():
    # Create dataset
    dataset = SimpleDataset(size=1000)
    
    # Create resumable dataloader
    dataloader = create_resumable_dataloader(
        dataset, 
        batch_size=32,
        shuffle=True,
        num_workers=2
    )
    
    # Simulate training
    print("Starting training...")
    for i, (data, labels) in enumerate(dataloader):
        print(f"Batch {i}: data shape {data.shape}, labels shape {labels.shape}")
        
        # Simulate checkpoint saving
        if i == 5:
            print("\nSaving checkpoint at batch 5...")
            state = dataloader.state_dict()
            
            # Simulate resume by creating new dataloader and loading state
            print("\nCreating new dataloader and resuming...")
            new_dataloader = create_resumable_dataloader(
                dataset, 
                batch_size=32,
                shuffle=True,
                num_workers=2
            )
            new_dataloader.load_state_dict(state)
            
            print("Resumed training:")
            for j, (data, labels) in enumerate(new_dataloader):
                print(f"Batch {j + i + 1}: data shape {data.shape}, labels shape {labels.shape}")
                if j >= 3:  # Just show a few more batches
                    break
            break


if __name__ == "__main__":
    main()