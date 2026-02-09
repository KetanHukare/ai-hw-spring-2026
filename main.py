
import torch
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset

from models import MLP, CNN, TransformerEncoder
from train import train_model


# Data Loading Functions
def get_transforms(augment=False):
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

    def __init__(self, split='train', transform=None):
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
    """Get MNIST train and test dataloaders from Hugging Face."""
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


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def plot_results(results, save_path='results.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for model_name, history in results.items():
        epochs = range(1, len(history['train_loss']) + 1)
        axes[0].plot(epochs, history['train_loss'], label=f'{model_name} (train)')
        axes[0].plot(epochs, history['test_loss'], '--', label=f'{model_name} (test)')
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Test Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    for model_name, history in results.items():
        epochs = range(1, len(history['train_acc']) + 1)
        axes[1].plot(epochs, history['train_acc'], label=f'{model_name} (train)')
        axes[1].plot(epochs, history['test_acc'], '--', label=f'{model_name} (test)')
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Test Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nResults plot saved to {save_path}")


def print_summary(best_accuracies):
    """Print summary table of results."""
    print("\n" + "=" * 50)
    print("FINAL RESULTS SUMMARY")
    print("=" * 50)
    print(f"{'Model':<20} {'Best Test Accuracy':>20}")
    print("-" * 50)
    for model_name, acc in best_accuracies.items():
        print(f"{model_name:<20} {acc:>19.2f}%")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description='MNIST Image Recognition with Multiple Models')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    parser.add_argument('--model', type=str, default='all', 
                        choices=['all', 'mlp', 'cnn', 'transformer'],
                        help='Which model to train')
    args = parser.parse_args()
    
    device = get_device()
    print(f"Using device: {device}")
    
    print(f"\nLoading MNIST dataset (augmentation: {args.augment})...")
    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        augment_train=args.augment
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    models_to_train = {}
    if args.model in ['all', 'mlp']:
        models_to_train['MLP'] = MLP()
    if args.model in ['all', 'cnn']:
        models_to_train['CNN'] = CNN()
    if args.model in ['all', 'transformer']:
        models_to_train['Transformer'] = TransformerEncoder()
    
    results = {}
    best_accuracies = {}
    
    for model_name, model in models_to_train.items():
        print(f"\n{'=' * 50}")
        print(f"Training {model_name}")
        print(f"{'=' * 50}")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        history, best_acc = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=args.epochs,
            lr=args.lr,
            device=device
        )
        
        results[model_name] = history
        best_accuracies[model_name] = best_acc
    
    print_summary(best_accuracies)
    
    if len(results) > 0:
        plot_results(results)


if __name__ == "__main__":
    main()
