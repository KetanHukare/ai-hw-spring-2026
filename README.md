# MNIST Image Recognition Neural Networks

A comparison of three neural network architectures for MNIST digit classification.

## Models

1. **MLP (Multi-Layer Perceptron)** - Shallow feedforward network with 2 hidden layers
2. **CNN (Convolutional Neural Network)** - 3 conv layers with max pooling
3. **Transformer Encoder** - Vision Transformer-style with patch embeddings and self-attention

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Train all models
```bash
python3 main.py
```

### Train a specific model
```bash
python3 main.py --model mlp
python3 main.py --model cnn
python3 main.py --model transformer
```

### With data augmentation (can improve test accuracy)
```bash
python3 main.py --augment
```

### Custom training parameters
```bash
python3 main.py --epochs 20 --batch-size 128 --lr 0.0005
```

## Data Augmentation Options

When `--augment` is enabled, training data is augmented with:
- Random rotation (±10°)
- Random affine transformations (translation, scaling)

## Output

- Training progress printed to console
- `results.png` - Plot comparing loss and accuracy curves for all models
