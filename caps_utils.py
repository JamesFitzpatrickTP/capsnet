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
    pad = 1
    padding = (pad, pad) * 2 + (0, 0) * (tensor.dim() - 2)
    tensor = fn.pad(tensor, padding, mode='constant', value=tensor.min())
    a, b = tensor.size(0), tensor.size(1)
    i_str, j_str, k_str, l_str = tensor.stride()
    new_shape = (a, b) + (out_dim, out_dim, ker_width, ker_width)
    tensor = tensor.as_strided(new_shape,
                               (i_str, j_str,
                                int(k_str * stride),
                                int(l_str * stride),
                                int(k_str * stride),
                                int(l_str * stride)))
    tensor_product = torch.einsum('abcdef,efbhi->acdhi', (tensor, weights))
    return tensor_product


def mat_conv_II(tensor, weights, ker_width, stride, in_dim, out_dim):
    a, b, c, d = tuple(tensor.size())
    new_tensor = torch.zeros((a, b, int(c * 2), int(d * 2)))
    new_tensor[:,:,::2,::2] = new_tensor[:,:,::2,::2].clone() + tensor
    pad = compute_padding(in_dim, out_dim, ker_width, stride)
    pad = 1
    padding = (pad, pad) * 2 + (0, 0) * (new_tensor.dim() - 2)
    tensor = fn.pad(new_tensor, padding, mode='constant', value=tensor.min())
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
