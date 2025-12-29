import torch
import torch.nn as nn

class DustCNN(nn.Module):
    """
    Convolutional Neural Network (CNN) for predicting dust extinction (tau) 
    from simulated galaxy images.

    Architecture:
    - 3 convolutional layers to extract hierarchical features
    - ReLU activations to introduce non-linearity
    - MaxPooling to reduce spatial dimensions
    - AdaptiveAvgPool2d to produce fixed-size features regardless of input size
    - Fully connected layer to output a single scalar (tau)
    """

    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            # First conv layer: 1 input channel (grayscale), 16 output channels, 3x3 kernel
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Reduces image size by factor of 2
            
            # Second conv layer: 16 -> 32 channels
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Further downsampling
            
            # Third conv layer: 32 -> 64 channels
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # Pools to 1x1 spatial size
            
            nn.Flatten(),             # Flatten 64x1x1 → 64
            nn.Linear(64, 1)          # Output layer: predicts single scalar
        )
    
    def forward(self, x):
        """
        Forward pass of the network.
        Input: x (batch_size, 1, H, W) grayscale images
        Output: predicted dust extinction tau
        """
        return self.net(x).squeeze()  # squeeze to remove singleton dimensions
