import torch
import torch.nn as nn
import torch.nn.functional as fn

import numpy as np


def compute_padding(in_dim, out_dim, ker_size, stride):
    numerator = ker_size - in_dim + stride * (out_dim - 1)
    return int(numerator / 2)


def set_padding(in_dim, out_dim, ker_size, stride, mode):
    if not isinstance(mode, str):
        raise TypeError("Padding must be a string of same, half or none")
    if mode == 'same':
        padding = compute_padding(in_dim, in_dim, ker_size, stride)
    elif mode == 'half':
        padding = compute_padding(in_dim, in_dim / 2, ker_size, stride) + 1
    else:
        padding = 0
    return padding


def reshape(tensor, shape):
    return torch.Tensor.view(tensor, shape)
    

class Convolve(nn.Module):
    def __init__(self, in_maps=1, out_maps=1, ker_size=3, stride=1, mode='none'):
        super(Convolve, self).__init__()

        self.in_maps = in_maps
        self.out_maps = out_maps
        self.ker_size = ker_size
        self.stride = stride

        self.padding = set_padding(in_maps, out_maps, ker_size, stride, mode)
        
        self.conv = nn.Conv2d(
            in_channels=self.in_maps,
            out_channels=self.out_maps,
            kernel_size=self.ker_size,
            stride=self.stride,
            padding=self.padding
        )

    def forward(self, tensor):
        return fn.relu(self.conv(tensor))


class Downsample(nn.Module):
    def __init__(self, in_maps=1, out_maps=1, ker_size=2, stride=2):
        super(Downsample, self).__init__()

        self.in_maps = in_maps
        self.out_maps = out_maps
        self.ker_size = ker_size
        self.stride = 2

        self.down = nn.Conv2d(
            in_channels=self.in_maps,
            out_channels=self.out_maps,
            kernel_szie=self.kernel_size,
            stride=self.stride
        )

    def forward(self, tensor):
        # Should there be a nonlinearity here?
        return self.down(tensor)
    

class InitialCaps(nn.Module):
    def __init__(self, in_maps, out_maps, ker_size, num_caps):
        super(InitialCaps, self).__init__()

        self.in_maps = in_maps
        self.out_maps = out_maps
        self.ker_size = ker_size
        self.num_caps = num_caps

        self.conv = Convolve(in_maps=self.in_maps,
                             out_maps=self.out_maps,
                             ker_size=self.ker_size,
                             mode='same')

    def forward(self, tensor):
        tensor = self.conv(tensor)
        tensor = self.squash(tensor)
        return tensor
        
    def squash(self, tensor):
        norm = torch.norm(tensor, dim=0)
        numerator = (norm ** 2) * tensor
        denominator = (1 + norm ** 2) * norm
        return numerator / denominator


class DownCaps(nn.Module):
    def __init__(self, in_maps, out_maps, ker_size, mode,
                 stride, num_caps, num_routes, num_atoms):
        super(DownCaps, self).__init__()

        self.in_maps = in_maps
        self.out_maps = out_maps
        self.ker_size = ker_size
        self.mode = mode
        self.stride = stride
        self.num_caps = num_caps
        self.num_routes = num_routes
        self.num_atoms = num_atoms

        self.ker_height = 3
        self.ker_width = 3

        self.convs = nn.ModuleList([
            Convolve(
                in_maps=self.in_maps,
                out_maps=self.out_maps,
                ker_size=self.ker_size,
                stride=self.stride,
                mode=self.mode) for _ in range(num_atoms)])

    def forward(self, tensor):
        tensors = [conv(tensor) for conv in self.convs]
        tensors = torch.stack(tensors, dim=1)
        tensors = torch.Tensor.squeeze(tensors)

    def squash(self, tensor):
        norm = torch.norm(tensor, dim=0)
        numerator = (norm ** 2) * tensor
        denominator = (1 + norm ** 2) * norm
        return numerator / denominator

    def routing(self, tensor, tensors):
        prev_dim, cur_dim = tensor.shape(1), tensors.shape(1)
        prev_atoms = tensor.shape(0)
        M = torch.randn(prev_atoms,
                        self.num_atoms,
                        self.ker_height,
                        self.ker_width,
                        prev_dim,
                        cur_dim
        )
        


def test_shape():
    x = torch.tensor(np.zeros((1, 1, 512, 512))).float()
    operation_one = Convolve(1, 16, 5, 1, 'same')
    y = operation_one(x)
    operation_two = DownCaps(16, 16, 5, 'half', 2, 256 * 256, 1, 2)
    z = operation_two(y)
    return z
