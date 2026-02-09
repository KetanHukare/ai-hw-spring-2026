
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset


def get_transforms(augment=False):
    """
    Get transforms for MNIST dataset.
    
    Args:
        augment: If True, apply data augmentation (rotation, affine, etc.)
    
    Returns:
        transform pipeline
    """
    if augment:
        transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    return transform


class HuggingFaceMNIST(Dataset):
    """Wrapper for Hugging Face MNIST dataset."""
    
    def __init__(self, split='train', transform=None):
        """
        Args:
            split: 'train' or 'test'
            transform: torchvision transforms to apply
        """
        self.dataset = load_dataset('ylecun/mnist', split=split)
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        return image, label


def get_dataloaders(batch_size=64, augment_train=False, num_workers=0):
    """
    Get MNIST train and test dataloaders from Hugging Face.
    
    Args:
        batch_size: Batch size for dataloaders
        augment_train: If True, apply augmentation to training data
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, test_loader
    """
    train_transform = get_transforms(augment=augment_train)
    test_transform = get_transforms(augment=False)
    
    train_dataset = HuggingFaceMNIST(split='train', transform=train_transform)
    test_dataset = HuggingFaceMNIST(split='test', transform=test_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders(batch_size=64, augment_train=True)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
