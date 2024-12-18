from typing import NamedTuple, Optional
import torch
import numpy as np
from torchvision import transforms

class WallSample(NamedTuple):
    states: torch.Tensor
    locations: torch.Tensor
    actions: torch.Tensor


class WallDataset:
    def __init__(
        self,
        data_path,
        probing=False,
        device="cuda",
        train=True,
    ):
        self.device = device
        self.states = np.load(f"{data_path}/states.npy", mmap_mode="r")
        self.actions = np.load(f"{data_path}/actions.npy")
        self.train = train
        if self.train:
            self.transform = transforms.Compose([
                transforms.RandomCrop((60, 60)),  # random crop
            ])
        else:
            pass
        if probing:
            self.locations = np.load(f"{data_path}/locations.npy")
        else:
            self.locations = None

    def __len__(self):
        return len(self.states)

    def __getitem__(self, i):
        states = torch.from_numpy(self.states[i]).float().to(self.device)
        actions = torch.from_numpy(self.actions[i]).float().to(self.device)

        if self.locations is not None:
            locations = torch.from_numpy(self.locations[i]).float().to(self.device)
        else:
            locations = torch.empty(0).to(self.device)

        return WallSample(states=states, locations=locations, actions=actions)


def create_wall_dataloader(
    data_path,
    probing=False,
    device="cuda",
    batch_size=16,
    train=True,
    train_ratio=1.0,
    seed=42,
):
    
    ds = WallDataset(
        data_path=data_path,
        probing=probing,
        device=device,
    )
    if train_ratio == 1.0:
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size,
            shuffle=train,
            drop_last=True,
            pin_memory=False,
        )
        return loader
    
    else:
        # Create the full dataset
        full_dataset = WallDataset(
            data_path=data_path,
            probing=probing,
            device=device,
        )
        
        # Calculate lengths for train and test splits
        dataset_size = len(full_dataset)
        train_size = int(train_ratio * dataset_size)
        test_size = dataset_size - train_size
        
        # Use random_split to create train and test datasets
        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, 
            [train_size, test_size],
            generator=torch.Generator().manual_seed(seed)
        )
        
        # Select the appropriate dataset based on train parameter
        
        # Create and return the dataloader
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train,
            drop_last=True,
            pin_memory=False,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=False,
        )
        
        return train_loader, test_loader