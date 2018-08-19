import torch
import torch.nn as nn
import torch.nn.functional as fn

import numpy as np


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


def slice_tensor(tensor, i, j, ker_width):
    slices = [slice(None)] * tensor.dim()
    slices[-2] = slice(i, i + ker_width)
    slices[-1] = slice(j, j + ker_width)
    return(tensor[slices])
    

def mat_conv_I(tensor, weights, ker_width, stride, in_dim, out_dim):
    pad = compute_padding(in_dim, out_dim, ker_width, stride)
    padding = (pad, pad) * 2 + (0, 0) * (tensor.dim() - 2)
    tensor = fn.pad(tensor, padding, mode='constant')
    a, b = tensor.size(0), tensor.size(1)
    i_str, j_str, k_str, l_str = tensor.stride()
    new_shape = (a, b) + (out_dim, out_dim, ker_width, ker_width)
    tensor = tensor.as_strided(new_shape,
                               (i_str, j_str,
                                int(k_str * stride),
                                int(l_str),
                                int(k_str * stride),
                                int(l_str)))
    tensor_product = torch.einsum('abcdef,efbhi->acdhi', (tensor, weights))
    return tensor_product


def mat_conv_II(tensor, weights, ker_width, stride, in_dim, out_dim):
    a, b, c, d = tuple(tensor.size())
    new_tensor = torch.zeros((a, b, int(c * 2), int(d * 2)))
    new_tensor[:,:,::2,::2] = tensor
    pad = compute_padding(in_dim, out_dim, ker_width, stride)
    padding = (pad, pad) * 2 + (0, 0) * (new_tensor.dim() - 2)
    tensor = fn.pad(new_tensor, padding, mode='constant')
    a, b = new_tensor.size(0), new_tensor.size(1)
    i_str, j_str, k_str, l_str = new_tensor.stride()
    new_shape = (a, b) + (out_dim * 2, out_dim * 2, ker_width, ker_width)
    new_tensor = new_tensor.as_strided(new_shape,
                               (i_str, j_str,
                                int(k_str * stride),
                                int(l_str * stride),
                                int(k_str * stride),
                                int(l_str * stride)))
    tensor_product = torch.einsum('abcdef,efbhi->acdhi', (new_tensor, weights))
    return tensor_product


class Convolve(nn.Module):
    def __init__(self, in_maps=1, out_maps=1,
                 ker_size=3, stride=1, mode='none'):
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

        self.weights = torch.rand(self.ker_height,
                       self.ker_width,
                       self.in_maps,
                       self.num_atoms,
                       self.out_maps).float()

    def forward(self, tensor):
        votes = self.voting(tensor, self.weights)
        votes = votes.permute(0, 3, 4, 1, 2)
        return self.routing(votes, self.num_routes)

    def squash(self, tensor, dim=0):
        norm = torch.norm(tensor, dim=dim)
        numerator = (norm ** 2) * tensor
        denominator = (1 + norm ** 2) * norm
        return numerator / denominator

    def voting(self, tensor, weights):
        tensor = mat_conv_I(tensor, weights, self.ker_width,
                            self.stride, tensor.size(-1),
                            int(tensor.size(-1) / self.stride))
        return tensor

    def routing(self, votes, num_routes=1):
        a, b, c, d, e = votes.shape
        logits = torch.zeros((a, b, d, e)).float()
        for routes in range(num_routes):
            logits = fn.softmax(logits, dim=0)
            preds = torch.einsum('abcde,abde->bcde', (votes, logits))
            preds = self.squash(preds, dim=0)
            logits += torch.einsum('abcde,bcde->abde', (votes, preds))
        return preds


class UpCaps(nn.Module):
    def __init__(self, in_maps, out_maps, ker_size, mode,
                 stride, num_caps, num_routes, num_atoms):
        super(UpCaps, self).__init__()

        self.in_maps = in_maps
        self.out_maps = out_maps
        self.ker_size = ker_size
        self.mode = mode
        self.stride = stride
        self.num_caps = num_caps
        self.num_routes = num_routes
        self.num_atoms = num_atoms

        self.ker_height = 5
        self.ker_width = 5

        self.convs = nn.ModuleList([
            Convolve(
                in_maps=self.in_maps,
                out_maps=self.out_maps,
                ker_size=self.ker_size,
                stride=self.stride,
                mode=self.mode) for _ in range(num_atoms)])

        self.weights = torch.randn(self.ker_height,
                       self.ker_width,
                       self.in_maps,
                       self.num_atoms,
                       self.out_maps).float()

    def forward(self, tensor):
        votes = self.voting(tensor, self.weights)
        votes = votes.permute(0, 3, 4, 1, 2)
        return self.routing(votes, self.num_routes)

    def squash(self, tensor, dim=0):
        norm = torch.norm(tensor, dim=dim)
        numerator = (norm ** 2) * tensor
        denominator = (1 + norm ** 2) * norm
        return numerator / denominator

    def voting(self, tensor, weights):
        tensor = mat_conv_II(tensor, weights, self.ker_width,
                          self.stride, tensor.size(-1),
                          int(tensor.size(-1) / self.stride))
        return tensor

    def routing(self, votes, num_routes=1):
        a, b, c, d, e = votes.shape
        logits = torch.zeros((a, b, d, e)).float()
        for routes in range(num_routes):
            logits = fn.softmax(logits, dim=0)
            preds = torch.einsum('abcde,abde->bcde', (votes, logits))
            preds = self.squash(preds, dim=0)
            logits += torch.einsum('abcde,bcde->abde', (votes, preds))
        return preds


def test_shape(in_tensor):
    convolution_one = Convolve(1, 16, 5, 1, 'same')
    a = convolution_one(in_tensor)
    downsampling_one = DownCaps(16, 16, 5, 'same', 2, 256 * 256, 1, 2)
    b = downsampling_one(a)
    capsuling_one = DownCaps(16, 16, 5, 'same', 1, 256 * 256, 3, 4)
    c = capsuling_one(b)
    downsampling_two = DownCaps(16, 32, 5, 'same', 2, 128 * 128, 3, 4)
    d = downsampling_two(c)
    capsuling_two = DownCaps(32, 32, 5, 'same', 1, 128 * 128, 3, 8)
    e = capsuling_two(d)
    downsampling_three = DownCaps(32, 64, 5, 'same', 2, 64 * 64, 3, 8)
    f = downsampling_three(e)
    capsuling_three = DownCaps(64, 64, 5, 'same', 1, 64 * 64, 3, 8)
    g = capsuling_three(f)
    capsuling_four = DownCaps(64, 32, 5, 'same', 1, 64 * 64, 3, 8)
    h = capsuling_four(g)
    upsampling_one = UpCaps(32, 32, 5, 'none', 1, 128 * 128, 3, 8)
    i = upsampling_one(h)
    capsuling_five = DownCaps(32, 32, 5, 'same', 1, 128 * 128, 3, 8)
    j = capsuling_five(torch.cat((e, i)))
    capsuling_six = DownCaps(32, 32, 5, 'same', 1, 128 * 128, 3, 4)
    k = capsuling_six(j)
    upsampling_two = UpCaps(32, 16, 5, 'none', 1, 256 * 256, 3, 8)
    l = upsampling_two(k)
    capsuling_seven = DownCaps(16, 16, 5, 'same', 1, 256 * 256, 3, 4)
    m = capsuling_seven(torch.cat((c, l)))
    capsuling_eight = DownCaps(16, 16, 5, 'same', 1, 256 * 256, 3, 4)
    n = capsuling_eight(m)
    upsampling_two = UpCaps(16, 16, 5, 'none', 1, 512 * 512, 3, 2)
    o = upsampling_two(n)
    capsuling_nine = DownCaps(16, 16, 5, 'same', 1, 512 * 512, 3, 2)
    p = capsuling_nine(torch.cat((a, o)))
    capsuling_ten = DownCaps(16, 16, 5, 'same', 1, 512 * 512, 3, 1)
    q = capsuling_ten(p)
    
    return q


import pydicom
import matplotlib.pyplot as plt
base_path = '/mnt/c/Users/James Fitzpatrick/Documents/P00019/manual_print_data/'
file_path = '/segmentation/ml/00_original/P00019_00_original.0052.dcm'
dcm = pydicom.read_file(base_path + file_path)
img = dcm.pixel_array
ten = np.expand_dims(img, 0)
ten = np.expand_dims(ten, 0)
ten = torch.tensor(ten).float()

net = test_shape(ten)
