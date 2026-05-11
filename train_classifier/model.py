import torch
import torch.nn as nn
import torch.nn.functional as functional

"""
Simple pocket classifier architecture. We apply two convolution layers to create hidden 64D feature arrays for each atom.
From here, we apply global max pooling to create a single 64D feature array representative of the entire pocket.
Lastly, we use two fully connected layers to create a binary output score between 0 and 1
"""

class PocketClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutions
        self.conv1 = nn.Conv1d(29, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 256, kernel_size=1)

        # Fully connected layers
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

        # Dropout
        self.dropout = nn.Dropout(p=0.15)

    def forward(self, x):
        x = functional.relu(self.conv1(x))
        x = functional.relu(self.conv2(x))

        # Global max pooling
        x = torch.max(x, dim=2).values

        x = self.dropout(functional.relu(self.fc1(x)))

        # Use sigmoid function to produce single score between 0 and 1
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze(1)
