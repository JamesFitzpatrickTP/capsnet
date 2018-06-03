import torch
from torchvision import datasets, transforms
import numpy as np


def load_images():
    mnist_train = datasets.MNIST(root='../mnist_data', train=True, download=True, transform=None)
    mnist_test = datasets.MNIST(root='../mnist_data', train=False, download=True, transform=None)
    return mnist_train, mnist_test


def mock_images():
    train, test = np.random.randn(10, 1, 28, 28), np.random.randn(10, 1, 28, 28)
    train, test = torch.tensor(train).float(), torch.tensor(test).float()
    return train, test

