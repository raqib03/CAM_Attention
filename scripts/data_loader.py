import torch
from torchvision import datasets, transforms
import os

def get_data_loaders(config):
    """Prepare train and test data loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize grayscale images
    ])
    
    train_dataset = datasets.FashionMNIST(
        root=config["dataset"]["root_dir"],
        train=True,
        transform=transform,
        download=True
    )
    
    test_dataset = datasets.FashionMNIST(
        root=config["dataset"]["root_dir"],
        train=False,
        transform=transform,
        download=True
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=2
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, test_loader