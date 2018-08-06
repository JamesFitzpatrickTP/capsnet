import torch
import torch.nn as nn
import torch.nn.functional as fn


def compute_padding(in_dim, out_dim, ker_size, stride):
    numerator = ker_size - in_dim + stride * (out_dim - 1)
    return int(numerator / 2)


def set_padding(in_dim, out_dim, ker_size, stride, mode):
    if not isinstance(mode, str):
        raise TypeError("Padding must be a string of either same or none")
    if mode == 'same':
        padding = compute_padding(in_dim, in_dim, ker_size, stride)
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

        print(self.padding)
        
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
    

class InitialCaps():
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
        return self.squash(self.conv(tensor))
        
    def squash(self, tensor):
        norm = torch.norm(tensor, dim=0)
        numerator = (norm ** 2) * tensor
        denominator = (1 + norm ** 2) * norm
        return numerator / denominator
