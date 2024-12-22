import torch
import torch.nn as nn

class CAMModel(nn.Module):
    def __init__(self, num_classes=10):
        super(CAMModel, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(3, 5, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample to 14x14
        
        self.conv3 = nn.Conv2d(5, 5, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        
        self.conv4 = nn.Conv2d(5, 3, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample to 7x7
        
        # Global Average Pooling (GAP)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # Output size: (num_channels, 1, 1)
        
        # Fully Connected Layer
        self.fc = nn.Linear(3, num_classes)
    
    def forward(self, x):
        # Convolutional Layers
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool1(x)
        
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.pool2(x)
        
        # Global Average Pooling
        x = self.gap(x)  # Shape: (batch_size, num_channels, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, num_channels)
        
        # Fully Connected Layer
        out = self.fc(x)
        return out
