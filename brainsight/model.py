import numpy as np
import torch.nn as nn

class CNNClassifier(nn.Module):
    def __init__(self, in_channels):
        super(CNNClassifier, self).__init__()
        # Define convolutional layers
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        # Define pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Define fully connected layers
        self.fc1 = nn.Linear(32 * 25, 128)
        self.fc2 = nn.Linear(128, 2)
        
        # Define activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Pass input through convolutional layers
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        # Flatten output of convolutional layers
        x = x.view(-1, 32 * 25)
        
        # Pass output through fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Apply softmax activation function for binary classification
        x = self.softmax(x)
        return x
