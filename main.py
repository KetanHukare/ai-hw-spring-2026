"""
Main script to train and compare all models on MNIST
"""
import torch
import argparse
import matplotlib.pyplot as plt

from data import get_dataloaders
from models import MLP, CNN, TransformerEncoder
from train import train_model


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def plot_results(results, save_path='results.png'):
    """Plot training curves for all models."""
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
