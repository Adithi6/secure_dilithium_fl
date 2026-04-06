# model/cnn.py
# Defines the neural network architecture used by every FL client.
# A small CNN for MNIST: 2 conv layers + 2 linear layers.

import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 28x28 -> 14x14

            nn.Conv2d(16, 32, kernel_size=3, padding=1), # 14x14 -> 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 14x14 -> 7x7

            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),                           # 10 MNIST classes
        )

    def forward(self, x):
        return self.net(x)
