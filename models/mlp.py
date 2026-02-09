"""
Shallow Multi-Layer Perceptron (MLP) for MNIST
"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    A shallow MLP with one hidden layer.
    Input: 28x28 = 784 flattened pixels
    Output: 10 classes (digits 0-9)
    """
    
    def __init__(self, input_size=784, hidden_size=256, num_classes=10, dropout=0.2):
        super(MLP, self).__init__()
        
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)


if __name__ == "__main__":
    model = MLP()
    print(model)
    
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
