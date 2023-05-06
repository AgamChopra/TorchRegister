# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 2023

@author: Agam Chopra
"""
import torch
import torch.nn as nn
from math import ceil
from numpy import flip, array


def norm(x):
    EPSILON = 1E-9
    try:
        return (x - torch.min(x)) / ((torch.max(x) - torch.min(x)) + EPSILON)
    except Exception:
            print('WARNING: Input could not be normalized!')


# N-D padding function of form (Batch, Channel,...)
def padNd(input_, target, device='cpu', mode='constant', value=0):
    dims = len(input_.shape) - 2
    delta = [target.shape[2+i] - input_.shape[2+i] for i in range(dims)]
    pads = tuple(flip(array(
        [[ceil(delta[i]/2), delta[i] - ceil(delta[i]/2)] for i in range(dims)]).flatten()))

    return nn.functional.pad(input=input_, pad=pads, mode=mode, value=value).to(dtype=torch.float, device=device)


class Theta(nn.Module):
    def __init__(self):
        super(Theta, self).__init__()
        self.sin = torch.sin
        self.cos = torch.cos

    def forward(self, x):
        output = x.clone()
        if output.shape[1] > 6:
            psi, theta, phi = x[:, 0], x[:, 1], x[:, 2]
            output[:, 0] = self.cos(psi) * self.cos(theta)
            output[:, 1] = self.sin(phi) * self.sin(psi) * \
                self.cos(theta) - self.cos(phi) * self.sin(theta)
            output[:, 2] = self.cos(phi) * self.sin(psi) * \
                self.cos(theta) + self.sin(phi) * self.sin(theta)

            output[:, 4] = self.cos(psi) * self.sin(theta)
            output[:, 5] = self.sin(phi) * self.sin(psi) * \
                self.sin(theta) + self.cos(phi) * self.cos(theta)
            output[:, 6] = self.cos(phi) * self.sin(psi) * \
                self.sin(theta) - self.sin(phi) * self.cos(theta)

            output[:, 8] = - self.sin(psi)
            output[:, 9] = self.sin(phi) * self.cos(psi)
            output[:, 10] = self.cos(phi) * self.cos(psi)
        else:
            theta = x[:, 0]
            output[:, 0] = self.cos(theta)
            output[:, 1] = - self.sin(theta)
            output[:, 3] = self.sin(theta)
            output[:, 4] = self.cos(theta)
        return output


class Regressor(nn.Module):
    def __init__(self, moving, per, device):
        super(Regressor, self).__init__()
        if len(moving.shape) == 5:
            self.reg = nn.Sequential(nn.Linear(int(2 * per * (torch.flatten(
                moving).shape[0])), 64, bias=False), nn.ReLU(), nn.Linear(64, 12)).to(device=device)
            self.reg[0].weight.data.zero_()
            self.reg[2].weight.data.zero_()
            self.reg[2].bias.data.copy_(torch.tensor(
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))
        else:
            self.reg = nn.Sequential(nn.Linear(int(2 * per * (torch.flatten(
                moving).shape[0])), 32, bias=False), nn.ReLU(), nn.Linear(32, 6)).to(device=device)
            self.reg[0].weight.data.zero_()
            self.reg[2].weight.data.zero_()
            self.reg[2].bias.data.copy_(torch.tensor(
                [1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.thetas = Theta()

    def forward(self, input_):
        var = self.reg(input_)
        theta = self.thetas(var)
        if theta.shape[-1] == 12:
            return theta.view(1, 3, 4)
        else:
            return theta.view(1, 2, 3)


class SpatialTransformer(nn.Module):
    '''
    N-D Spatial Transformer
    Obtained from https://github.com/voxelmorph/voxelmorph
    '''

    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.mode = mode

        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * \
                (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nn.functional.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class attention_grid(nn.Module):
    def __init__(self, x_c, g_c, i_c, stride=3, mode='nearest', dims=3):
        super(attention_grid, self).__init__()
        if dims == 3:
            self.input_filter = nn.Conv3d(
                in_channels=x_c, out_channels=i_c, kernel_size=1, stride=stride, bias=False)
            self.gate_filter = nn.Conv3d(
                in_channels=g_c, out_channels=i_c, kernel_size=1, stride=1, bias=True)
            self.psi = nn.Conv3d(in_channels=i_c, out_channels=1,
                                 kernel_size=1, stride=1, bias=True)
            self.bnorm = nn.InstanceNorm3d(i_c)
        else:
            self.input_filter = nn.Conv2d(
                in_channels=x_c, out_channels=i_c, kernel_size=1, stride=stride, bias=False)
            self.gate_filter = nn.Conv2d(
                in_channels=g_c, out_channels=i_c, kernel_size=1, stride=1, bias=True)
            self.psi = nn.Conv2d(in_channels=i_c, out_channels=1,
                                 kernel_size=1, stride=1, bias=True)
            self.bnorm = nn.InstanceNorm2d(i_c)
        self.mode = mode

    def forward(self, x, g, device):
        x_shape = x.shape

        a = self.input_filter(x)
        b = self.gate_filter(g)

        if a.shape[-1] < b.shape[-1]:
            a = padNd(a, b, device)

        elif a.shape[-1] > b.shape[-1]:
            b = padNd(b, a, device)

        w = torch.sigmoid(self.psi(nn.functional.relu(a + b)))
        w = nn.functional.interpolate(w, size=x_shape[2:], mode=self.mode)

        y = x * w
        y = self.bnorm(y)
        return y, w


class Attention_UNet(nn.Module):
    def __init__(self, img_size, mode='nearest', in_c=1, n=1):
        super(Attention_UNet, self).__init__()
        if len(img_size) == 3:
            out_c = 3
            self.layer1 = nn.Sequential(nn.Conv3d(in_channels=in_c, out_channels=int(64/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm3d(int(64/n)),
                                        nn.Conv3d(in_channels=int(64/n), out_channels=int(64/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm3d(int(64/n)))

            self.skip1 = attention_grid(
                int(64/n), int(64/n), int(64/n), dims=3)

            self.layer2 = nn.Sequential(nn.Conv3d(in_channels=int(64/n), out_channels=int(128/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm3d(int(128/n)),
                                        nn.Conv3d(in_channels=int(128/n), out_channels=int(128/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm3d(int(128/n)))

            self.skip2 = attention_grid(
                int(128/n), int(128/n), int(128/n), dims=3)

            self.layer3 = nn.Sequential(nn.Conv3d(in_channels=int(128/n), out_channels=int(256/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm3d(int(256/n)),
                                        nn.Conv3d(in_channels=int(256/n), out_channels=int(256/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm3d(int(256/n)))

            self.skip3 = attention_grid(
                int(256/n), int(256/n), int(256/n), dims=3)

            self.layer4 = nn.Sequential(nn.Conv3d(in_channels=int(256/n), out_channels=int(512/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm3d(int(512/n)),
                                        nn.Conv3d(in_channels=int(512/n), out_channels=int(512/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm3d(int(512/n)))

            self.skip4 = attention_grid(
                int(512/n), int(512/n), int(512/n), dims=3)

            self.layer5 = nn.Sequential(nn.Conv3d(in_channels=int(512/n), out_channels=int(1024/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm3d(int(1024/n)),
                                        nn.Conv3d(in_channels=int(
                                            1024/n), out_channels=int(1024/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm3d(int(1024/n)),
                                        nn.ConvTranspose3d(in_channels=int(1024/n), out_channels=int(512/n), kernel_size=2, stride=2), nn.ReLU(), nn.InstanceNorm3d(int(512/n)))

            self.layer6 = nn.Sequential(nn.Conv3d(in_channels=int(1024/n), out_channels=int(512/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm3d(int(512/n)),
                                        nn.Conv3d(in_channels=int(
                                            512/n), out_channels=int(512/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm3d(int(512/n)),
                                        nn.ConvTranspose3d(in_channels=int(512/n), out_channels=int(256/n), kernel_size=2, stride=2), nn.ReLU(), nn.InstanceNorm3d(int(256/n)))

            self.layer7 = nn.Sequential(nn.Conv3d(in_channels=int(512/n), out_channels=int(256/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm3d(int(256/n)),
                                        nn.Conv3d(in_channels=int(
                                            256/n), out_channels=int(256/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm3d(int(256/n)),
                                        nn.ConvTranspose3d(in_channels=int(256/n), out_channels=int(128/n), kernel_size=2, stride=2), nn.ReLU(), nn.InstanceNorm3d(int(128/n)))

            self.layer8 = nn.Sequential(nn.Conv3d(in_channels=int(256/n), out_channels=int(128/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm3d(int(128/n)),
                                        nn.Conv3d(in_channels=int(
                                            128/n), out_channels=int(128/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm3d(int(128/n)),
                                        nn.ConvTranspose3d(in_channels=int(128/n), out_channels=int(64/n), kernel_size=2, stride=2), nn.ReLU(), nn.InstanceNorm3d(int(64/n)))

            self.layer9 = nn.Sequential(nn.Conv3d(in_channels=int(128/n), out_channels=int(64/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm3d(int(64/n)),
                                        nn.Conv3d(in_channels=int(64/n), out_channels=int(64/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm3d(int(64/n)))

            self.out = nn.Conv3d(in_channels=int(
                64/n), out_channels=out_c, kernel_size=1)

            self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        else:
            out_c = 2
            self.layer1 = nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=int(64/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm2d(int(64/n)),
                                        nn.Conv2d(in_channels=int(64/n), out_channels=int(64/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm2d(int(64/n)))

            self.skip1 = attention_grid(
                int(64/n), int(64/n), int(64/n), dims=2)

            self.layer2 = nn.Sequential(nn.Conv2d(in_channels=int(64/n), out_channels=int(128/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm2d(int(128/n)),
                                        nn.Conv2d(in_channels=int(128/n), out_channels=int(128/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm2d(int(128/n)))

            self.skip2 = attention_grid(
                int(128/n), int(128/n), int(128/n), dims=2)

            self.layer3 = nn.Sequential(nn.Conv2d(in_channels=int(128/n), out_channels=int(256/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm2d(int(256/n)),
                                        nn.Conv2d(in_channels=int(256/n), out_channels=int(256/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm2d(int(256/n)))

            self.skip3 = attention_grid(
                int(256/n), int(256/n), int(256/n), dims=2)

            self.layer4 = nn.Sequential(nn.Conv2d(in_channels=int(256/n), out_channels=int(512/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm2d(int(512/n)),
                                        nn.Conv2d(in_channels=int(512/n), out_channels=int(512/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm2d(int(512/n)))

            self.skip4 = attention_grid(
                int(512/n), int(512/n), int(512/n), dims=2)

            self.layer5 = nn.Sequential(nn.Conv2d(in_channels=int(512/n), out_channels=int(1024/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm2d(int(1024/n)),
                                        nn.Conv2d(in_channels=int(
                                            1024/n), out_channels=int(1024/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm2d(int(1024/n)),
                                        nn.ConvTranspose2d(in_channels=int(1024/n), out_channels=int(512/n), kernel_size=2, stride=2), nn.ReLU(), nn.InstanceNorm2d(int(512/n)))

            self.layer6 = nn.Sequential(nn.Conv2d(in_channels=int(1024/n), out_channels=int(512/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm2d(int(512/n)),
                                        nn.Conv2d(in_channels=int(
                                            512/n), out_channels=int(512/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm2d(int(512/n)),
                                        nn.ConvTranspose2d(in_channels=int(512/n), out_channels=int(256/n), kernel_size=2, stride=2), nn.ReLU(), nn.InstanceNorm2d(int(256/n)))

            self.layer7 = nn.Sequential(nn.Conv2d(in_channels=int(512/n), out_channels=int(256/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm2d(int(256/n)),
                                        nn.Conv2d(in_channels=int(
                                            256/n), out_channels=int(256/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm2d(int(256/n)),
                                        nn.ConvTranspose2d(in_channels=int(256/n), out_channels=int(128/n), kernel_size=2, stride=2), nn.ReLU(), nn.InstanceNorm2d(int(128/n)))

            self.layer8 = nn.Sequential(nn.Conv2d(in_channels=int(256/n), out_channels=int(128/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm2d(int(128/n)),
                                        nn.Conv2d(in_channels=int(
                                            128/n), out_channels=int(128/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm2d(int(128/n)),
                                        nn.ConvTranspose2d(in_channels=int(128/n), out_channels=int(64/n), kernel_size=2, stride=2), nn.ReLU(), nn.InstanceNorm2d(int(64/n)))

            self.layer9 = nn.Sequential(nn.Conv2d(in_channels=int(128/n), out_channels=int(64/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm2d(int(64/n)),
                                        nn.Conv2d(in_channels=int(64/n), out_channels=int(64/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm2d(int(64/n)))

            self.out = nn.Conv2d(in_channels=int(
                64/n), out_channels=out_c, kernel_size=1)

            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.warp = SpatialTransformer(img_size, mode)

    def forward(self, x, device, out_att=False):
        y1 = self.layer1(x)
        y = self.maxpool(y1)

        y2 = self.layer2(y)
        y = self.maxpool(y2)

        y3 = self.layer3(y)
        y = self.maxpool(y3)

        y4 = self.layer4(y)
        y = self.maxpool(y4)

        y = self.layer5(y)
        y4, _ = self.skip4(y4, y, device=device)

        y = torch.cat((y4, padNd(y, y4, device=device)), dim=1)
        y = self.layer6(y)
        y3, _ = self.skip3(y3, y, device=device)

        y = torch.cat((y3, padNd(y, y3, device=device)), dim=1)
        y = self.layer7(y)
        y2, _ = self.skip2(y2, y, device=device)

        y = torch.cat((y2, padNd(y, y2, device=device)), dim=1)
        y = self.layer8(y)
        y1, _ = self.skip1(y1, y, device=device)

        y = torch.cat((y1, padNd(y, y1, device=device)), dim=1)
        y = self.layer9(y)

        y = padNd(y, x, device=device)

        flow = self.out(y)

        y = self.warp(x, flow)

        return y, flow
