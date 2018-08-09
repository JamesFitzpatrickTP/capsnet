import torch
import torch.nn as nn
import torch.nn.functional as fn

import numpy as np
from functools import partial


def axis_string(max_dims):
    strings = (chr(97 + i) for i in range(max_dims))
    string = ''.join(strings)
    return string


def index_string(string, indices):
    string = np.array(list(string))
    string = string[indices]
    return ''.join(string)


def ein_string(string, substring):
    indices = [(substring[i] in string) for i in range(len(substring))]
    indices = ~ np.array(indices).astype(bool)
    indices = list(indices) + [True] * int(len(string) - len(substring))
    indices = np.array(indices)
    return index_string(string, indices)


def parse_axes(string, axes):
    np_axes = np.array(axes)
    string_a = index_string(string, np_axes[:, 0]) 
    string_b = index_string(string, np_axes[:, 1])
    string_a = ein_string(string, string_a)
    string_b = ein_string(string, string_b)
    return string_a + string_b


def tensordot(tensor, weights, axes):
    max_dims = max(tensor.dim(), weights.dim())
    string = axis_string(max_dims)
    string_c = ''.join(sorted(set(parse_axes(string, axes))))
    string_a, string_b = string[:tensor.dim()], string[:weights.dim()]
    string = string_a + ',' + string_b + '->' + string_c
    print(string)
    return torch.einsum(string, (tensor, weights))


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
        self.routing(tensor, tensors)

    def squash(self, tensor):
        norm = torch.norm(tensor, dim=0)
        numerator = (norm ** 2) * tensor
        denominator = (1 + norm ** 2) * norm
        return numerator / denominator

    def routing(self, tensor, tensors):
        prev_dim, cur_dim = tensor.size(1), tensors.size(1)
        prev_atoms = tensor.size(0)
        M = torch.randn(prev_atoms,
                        self.ker_height,
                        self.ker_width,
                        prev_dim,
                        cur_dim,
                        self.num_atoms)


def test_shape():
    x = torch.tensor(np.zeros((1, 1, 512, 512))).float()
    operation_one = Convolve(1, 16, 5, 1, 'same')
    y = operation_one(x)
    operation_two = DownCaps(16, 16, 5, 'half', 2, 256 * 256, 1, 2)
    z = operation_two(y)
    return z


def slice_tensor(tensor, i, j, ker_width):
    slices = [slice(None)] * tensor.dim()
    slices[-2] = slice(i, i + ker_width)
    slices[-1] = slice(j, j + ker_width)
    return(tensor[slices])


def mat_conv(tensor, weights, ker_width, stride, in_dim, out_dim):
    pad = compute_padding(in_dim, out_dim, ker_width, stride)
    padding = (pad, pad) * 2 + (0, 0) * (tensor.dim() - 2) 
    tensor = fn.pad(tensor, padding, mode='constant')
    rng = in_dim - ker_width + 1
    test_tensor = slice_tensor(tensor, 0, 0, ker_width)
    print(test_tensor.shape, weights.shape)
    tensor_product = torch.einsum('abcd,dcbef->aef', (test_tensor, weights))
    return tensor_product
    
    
    
