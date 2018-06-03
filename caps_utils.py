import torch
import torch.nn as nn
import numpy as np
import load_mnist


def squash(vector, epsilon=1e-4):
    squared_norm = (vector ** 2).sum(-1, keepdim=True)
    normed_vector = vector / (torch.sqrt(squared_norm) + epsilon)
    prefactor =  squared_norm / (1 + squared_norm)
    return prefactor * normed_vector


def torch_tensor(shape):
    return torch.tensor(np.random.randn(shape)).float()


def convolve(in_maps, out_maps, kernel_size, stride=1, padding=1):
    return nn.Conv2d(
        in_channels=in_maps,
        out_channels=out_maps,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding)


class ConvConv(nn.Module):
    def __init__(self, in_maps, out_maps, kernel_size, strides):
        super(ConvConv, self).__init__()
        
        self.in_maps = in_maps
        self.out_maps = out_maps
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv_a = convolve(self.in_maps, self.out_maps, self.kernel_size, self.strides[0])
        self.conv_b = convolve(self.out_maps, self.out_maps, self.kernel_size, self.strides[1])
        
    def forward(self, tensor):
        print(tensor.shape)
        tensor = nn.functional.relu(self.conv_a(tensor))
        print(tensor.shape)
        tensor = nn.functional.relu(self.conv_b(tensor))
        print(tensor.shape)
        return tensor


class CapsLayer(nn.Module):
    def __init__(self, in_dim=None, out_dim=None, in_num=None, out_num=None):
        super(CapsLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_num = in_num
        self.out_num = out_num

        self.squash = squash

    def flatten(self, tensor, dim):
        batch_size = tensor.shape[0]
        tensor = tensor.view(batch_size, -1, dim)
        return tensor

    def forward(self, tensor):
        if self.in_dim is None:
            tensor = self.flatten(tensor, self.out_dim)
            print(tensor.shape)
        else:
            self.nums = self.in_num * self.out_num
            self.weights = torch_tensor((self.in_dim, self.out_dim, self.out_num))
            
        return tensor
    

class CapsNet(nn.Module):
    def __init__(self, in_maps, out_maps, kernel_size, strides):
        super(CapsNet, self).__init__()

        self.in_maps = in_maps
        self.out_maps = out_maps
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv = ConvConv(self.in_maps, self.out_maps, self.kernel_size, self.strides)
        self.in_caps = CapsLayer(None, 8)
        self.out_caps = CapsLayer(8, 16, 3, 2)
                
    def forward(self, tensor):
        tensor = nn.functional.relu(self.conv(tensor))
        tensor = self.in_caps(tensor)
        tensor = self.out_caps(tensor)
                            
        return tensor
