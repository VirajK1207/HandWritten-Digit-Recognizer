# src/data.py - Data Layer
# Downloads, augments and loads MNIST dataset

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import yaml
import os


def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


def get_transforms(augment=True):
    """
    Returns image transforms pipeline.
    Training uses augmentation, validation/test does not.
    """
    config = load_config()

    # Base transforms applied to all data
    base_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        # These are the official MNIST mean and std values
    ]

    if augment and config["augmentation"]["enabled"]:
        # Add augmentation for training data only
        aug_transforms = [
            transforms.RandomRotation(
                degrees=config["augmentation"]["rotation_degrees"]
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(
                    config["augmentation"]["shift_percent"],
                    config["augmentation"]["shift_percent"]
                )
            ),
        ]
        return transforms.Compose(aug_transforms + base_transforms)

    return transforms.Compose(base_transforms)


def get_dataloaders():
    """
    Downloads MNIST and returns train, validation and test dataloaders.

    Returns:
        train_loader  - augmented training data
        val_loader    - clean validation data
        test_loader   - clean test data
    """
    config = load_config()

    data_dir   = config["paths"]["data_dir"]
    batch_size = config["training"]["batch_size"]
    val_split  = config["training"]["validation_split"]
    seed       = config["training"]["seed"]

    os.makedirs(data_dir, exist_ok=True)

    print("Downloading MNIST dataset... (auto downloads ~11MB)")

    # Training data with augmentation
    train_data = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=get_transforms(augment=True)
    )

    # Test data without augmentation
    test_data = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=get_transforms(augment=False)
    )

    # Split training into train and validation
    val_size   = int(len(train_data) * val_split)
    train_size = len(train_data) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        train_data,
        [train_size, val_size],
        generator=generator
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"Train samples      : {train_size}")
    print(f"Validation samples : {val_size}")
    print(f"Test samples       : {len(test_data)}")

    return train_loader, val_loader, test_loader


def get_class_names():
    return [str(i) for i in range(10)]