import torch
import torch.nn as nn
import torch.nn.functional as fn

import caps_utils


class Convolve(nn.Module):
    def __init__(self, in_maps=1, out_maps=1,
                 ker_size=3, stride=1, mode='none'):
        super(Convolve, self).__init__()

        self.in_maps = in_maps
        self.out_maps = out_maps
        self.ker_size = ker_size
        self.stride = stride

        self.padding = caps_utils.set_padding(
            in_maps, out_maps, ker_size, stride, mode)
        
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

    def squash(self, tensor, dim=0, epsilon=1e-4):
        squared_norm = (tensor ** 2).sum(1, keepdim=True) + epsilon
        norm = torch.sqrt(squared_norm)
        numerator = squared_norm * tensor
        denominator = (1 + squared_norm) * norm 
        return numerator / denominator

    def voting(self, tensor, weights):
        tensor = caps_utils.mat_conv_I(tensor, weights, self.ker_width,
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
            logits = logits + torch.einsum('abcde,bcde->abde', (votes, preds))
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

    def squash(self, tensor, dim=0, epsilon=1e-4):
        squared_norm = (tensor ** 2).sum(1, keepdim=True) + epsilon
        norm = torch.sqrt(squared_norm)
        numerator = squared_norm * tensor
        denominator = (1 + squared_norm) * norm 
        return numerator / denominator

    def voting(self, tensor, weights):
        tensor = caps_utils.mat_conv_II(tensor, weights, self.ker_width,
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
            logits = logits + torch.einsum('abcde,bcde->abde', (votes, preds))
        return preds


class CapsuleNetwork(nn.Module):
    def __init__(self, ops, in_dims, out_dims, strides, routes, atoms, cats):
        super(CapsuleNetwork, self).__init__()

        self.ops = ops
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.strides = strides
        self.routes = routes
        self.atoms = atoms
        self.cats = cats

        self.ordered_ops = []

        for op, in_dim, out_dim, stride, route, atom in zip(
                self.ops,
                self.in_dims,
                self.out_dims,
                self.strides,
                self.routes,
                self.atoms):
            if op == Convolve:
                self.ordered_ops = self.ordered_ops.append(
                    op(in_dim, out_dim, 5, stride, 'same'))
            if op == DownCaps:
                self.ordered_ops = self.ordered_ops.append(
                    op(in_dim, out_dim, 5, 'same', stride, 0, route, atom))
            if op == UpCaps:
                self.ordered_ops = self.ordered_ops.append(
                    op(in_dim, out_dim, 5, 'same', stride, 0, route, atom))

    
class TestNetwork(nn.Module):
    def __init__(self):
        super(TestNetwork, self).__init__()

        self.convolution_one = Convolve(1, 8, 5, 1, 'same')
        self.downsampling_one = DownCaps(8, 8, 5, 'same', 2, 256 * 256, 1, 2)
        self.capsuling_one = DownCaps(8, 8, 5, 'same', 1, 256 * 256, 3, 4)
        self.downsampling_two = DownCaps(8, 16, 5, 'same', 2, 128 * 128, 3, 4)
        self.capsuling_two = DownCaps(16, 16, 5, 'same', 1, 128 * 128, 3, 8)
        self.downsampling_three = DownCaps(16, 32, 5, 'same', 2, 64 * 64, 3, 8)
        self.capsuling_three = DownCaps(32, 32, 5, 'same', 1, 64 * 64, 3, 8)
        self.capsuling_four = DownCaps(32, 16, 5, 'same', 1, 64 * 64, 3, 8)
        self.upsampling_one = UpCaps(16, 16, 5, 'none', 1, 128 * 128, 3, 8)
        self.capsuling_five = DownCaps(16, 16, 5, 'same', 1, 128 * 128, 3, 8)
        self.capsuling_six = DownCaps(16, 16, 5, 'same', 1, 128 * 128, 3, 4)
        self.upsampling_two = UpCaps(16, 8, 5, 'none', 1, 256 * 256, 3, 8)
        self.capsuling_seven = DownCaps(8, 8, 5, 'same', 1, 256 * 256, 3, 4)
        self.capsuling_eight = DownCaps(8, 8, 5, 'same', 1, 256 * 256, 3, 4)
        self.upsampling_three = UpCaps(8, 8, 5, 'none', 1, 512 * 512, 3, 2)
        self.capsuling_nine = DownCaps(8, 8, 5, 'same', 1, 512 * 512, 3, 2)
        self.capsuling_ten = DownCaps(8, 1, 5, 'same', 1, 512 * 512, 3, 1)

    def forward(self, tensor):
        a = self.convolution_one(tensor)
        b = self.downsampling_one(a)
        c = self.capsuling_one(b)
        d = self.downsampling_two(c)
        e = self.capsuling_two(d)
        f = self.downsampling_three(e)
        g = self.capsuling_three(f)
        h = self.capsuling_four(g)
        i = self.upsampling_one(h)
        j = self.capsuling_five(torch.cat((e, i)))
        k = self.capsuling_six(j)
        ll = self.upsampling_two(k)
        m = self.capsuling_seven(torch.cat((c, ll)))
        n = self.capsuling_eight(m)
        o = self.upsampling_three(n)
        p = self.capsuling_nine(torch.cat((a, o)))
        q = self.capsuling_ten(p)
        
        return q
