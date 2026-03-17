# src/model.py - Model Layer
# CNN architecture for MNIST digit recognition

import torch
import torch.nn as nn
import yaml

# Load config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)


class MNISTNet(nn.Module):
    """
    CNN Architecture for MNIST Digit Recognition

    Architecture:
        Conv Block 1: Conv2d -> BatchNorm -> ReLU -> MaxPool -> Dropout
        Conv Block 2: Conv2d -> BatchNorm -> ReLU -> MaxPool -> Dropout
        Classifier  : Flatten -> Linear -> ReLU -> Dropout -> Linear -> Softmax

    Why CNN over ANN?
        - CNNs use convolutional filters to detect local patterns (edges, curves)
        - ANNs treat every pixel independently - loses spatial information
        - CNNs share weights across image - fewer parameters, less overfitting
        - CNNs achieve 99%+ on MNIST, ANNs typically reach 97-98%
    """

    def __init__(self):
        super(MNISTNet, self).__init__()

        # Conv Block 1
        # Input: (batch, 1, 28, 28)
        # Output: (batch, 32, 12, 12)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            # BatchNorm normalizes outputs - speeds up training
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # MaxPool reduces spatial size by half
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Dropout randomly turns off neurons - prevents overfitting
            nn.Dropout2d(p=0.25)
        )

        # Conv Block 2
        # Input: (batch, 32, 14, 14)
        # Output: (batch, 64, 6, 6)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25)
        )

        # Classifier
        # Input: flattened conv output
        # Output: 10 class probabilities
        self.classifier = nn.Sequential(
            # Flatten 3D tensor to 1D
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            # Dropout for regularization
            nn.Dropout(p=0.5),
            nn.Linear(256, config["model"]["num_classes"])
            # No softmax here - CrossEntropyLoss applies it internally
        )

    def forward(self, x):
        """Forward pass through the network"""
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x


class SimpleANN(nn.Module):
    """
    Simple ANN for comparison with CNN.
    Used in evaluate.py to show why CNN is better.

    Architecture:
        Flatten -> Linear -> ReLU -> Dropout -> Linear -> ReLU -> Linear
    """

    def __init__(self):
        super(SimpleANN, self).__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, config["model"]["num_classes"])
        )

    def forward(self, x):
        return self.network(x)


def get_model():
    """Returns CNN model instance"""
    return MNISTNet()


def get_ann_model():
    """Returns ANN model instance for comparison"""
    return SimpleANN()


def count_parameters(model):
    """Counts total trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model with dummy input
    model = MNISTNet()
    dummy = torch.randn(1, 1, 28, 28)
    output = model(dummy)

    print("CNN Model Architecture:")
    print(model)
    print(f"\nInput shape  : {dummy.shape}")
    print(f"Output shape : {output.shape}")
    print(f"Parameters   : {count_parameters(model):,}")

    ann = SimpleANN()
    print(f"\nANN Parameters: {count_parameters(ann):,}")
    print(f"CNN Parameters: {count_parameters(model):,}")
    print("CNN uses fewer parameters but achieves higher accuracy!")