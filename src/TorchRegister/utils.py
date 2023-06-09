# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 2023

@author: Agam Chopra
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from math import ceil
from numpy import flip, array
import numpy as np


EPSILON = 1E-10


def K_gauss(input_):
    output_ = (1/(2 * torch.pi)) * torch.exp(-(input_ ** 2) / 2)

    return output_


def PDF_xis(signals, xis, h=3):  # h is the bandwidth
    x_diff = torch.stack([signals - xis[:, i:i+1]
                         for i in range(xis.shape[1])], dim=-1)

    x_ = x_diff / h
    tf = K_gauss(x_)
    p_xi = (1 / h) * torch.mean(tf, dim=1)

    return p_xi


def PDF(signals, Xs, h=3):
    pdf = PDF_xis(signals, Xs, h)

    return pdf


def get_pdf(data, steps=256, bandwidth=2):
    # [N,...] N flattened 1-d signals
    signals = torch.flatten(data, start_dim=1)

    min_val, max_val = torch.max(signals).item(), torch.min(
        signals).item()

    line_samples = torch.linspace(min_val, max_val, steps, dtype=torch.float, device=signals.device,
                                  requires_grad=False) * torch.ones((len(data), steps), dtype=torch.float, device=signals.device, requires_grad=False)

    pdf = PDF(signals, line_samples, h=bandwidth)

    return pdf


def NMI(img1, img2, bins=256, bandwidth=0.1):
    # Calculate the approx. histograms via PDF using KDE
    hist1 = get_pdf(img1, steps=bins, bandwidth=bandwidth)

    hist2 = get_pdf(img2, steps=bins, bandwidth=bandwidth)

    hist_joint = get_pdf(torch.stack((img1, img2), dim=1),
                         steps=bins, bandwidth=bandwidth)

    # Calculate the probability distributions
    p1 = hist1 / torch.unsqueeze(torch.sum(hist1, dim=1), dim=1)
    p2 = hist2 / torch.unsqueeze(torch.sum(hist2, dim=1), dim=1)
    p_joint = hist_joint / torch.unsqueeze(torch.sum(hist_joint, dim=1), dim=1)

    # Calculate the shannon entropy
    E1 = -torch.sum((p1 * -torch.log2(p1 + EPSILON)), dim=1)
    E2 = -torch.sum((p2 * -torch.log2(p2 + EPSILON)), dim=1)
    E_joint = -torch.sum((p_joint * -torch.log2(p_joint + EPSILON)), dim=1)

    # Calculate the mutual information
    mutual_info = E1 + E2 - E_joint
    norm_mutual_info = 2 * mutual_info / (E1 + E2)

    return norm_mutual_info, mutual_info


def get_sobel_kernel3D(n1=1, n2=2, n3=2):
    '''
    Returns 3D Sobel kernels Sx, Sy, Sz, and diagonal kernels.
    ex:
        Sx = [[[-n1, 0, n1],
               [-n2, 0, n2],
               [-n1, 0, n1]],
              [[-n2, 0, n2],
               [-n2*n3, 0, n2*n3],
               [-n2, 0, n2]],
              [[-n1, 0, n1],
               [-n2, 0, n2],
               [-n1, 0, n1]]]


    Parameters
    ----------
    n1 : int, optional
        kernel value 1. The default is 1.
    n2 : int, optional
        kernel value 2. The default is 2.
    n3 : int, optional
        kernel value 3. The default is 2.

    Returns
    -------
    list
        list of all the 3d sobel kernels.

    '''
    Sx = np.asarray([[[-n1, 0, n1], [-n2, 0, n2], [-n1, 0, n1]], [[-n2, 0, n2],
                    [-n3*n2, 0, n3*n2], [-n2, 0, n2]], [[-n1, 0, n1], [-n2, 0, n2], [-n1, 0, n1]]])
    Sy = np.asarray([[[-n1, -n2, -n1], [0, 0, 0], [n1, n2, n1]], [[-n2, -n3*n2, -n2],
                    [0, 0, 0], [n2, n3*n2, n2]], [[-n1, -n2, -n1], [0, 0, 0], [n1, n2, n1]]])
    Sz = np.asarray([[[-n1, -n2, -n1], [-n2, -n3*n2, -n2], [-n1, -n2, -n1]],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[n1, n2, n1], [n2, n3*n2, n2], [n1, n2, n1]]])
    Sd11 = np.asarray([[[0, n1, n2], [-n1, 0, n1], [-n2, -n1, 0]], [[0, n2, n2*n3],
                      [-n2, 0, n2], [-n2*n3, -n2, 0]], [[0, n1, n2], [-n1, 0, n1], [-n2, -n1, 0]]])
    Sd12 = np.asarray([[[-n2, -n1, 0], [-n1, 0, n1], [0, n1, n2]], [[-n2*n3, -n2, 0],
                      [-n2, 0, n2], [0, n2, n2*n3]], [[-n2, -n1, 0], [-n1, 0, n1], [0, n1, n2]]])
    Sd21 = Sd11.T
    Sd22 = Sd12.T
    Sd31 = np.asarray([-S.T for S in Sd11.T])
    Sd32 = np.asarray([S.T for S in Sd12.T])

    return [Sx, Sy, Sz, Sd11, Sd12, Sd21, Sd22, Sd31, Sd32]


class Edge3D():
    '''
    Sobel edge filter.
    '''

    def __init__(self, n1=1, n2=2, n3=2, device='cpu'):
        super(Edge3D, self).__init__()
        self.device = device
        k_sobel = 3
        S = get_sobel_kernel3D(n1, n2, n3)
        self.sobel_filters = []

        for s in S:
            sobel_filter = nn.Conv3d(in_channels=1, out_channels=1, stride=1,
                                     kernel_size=k_sobel, padding=k_sobel // 2, bias=False)
            sobel_filter.weight.data = torch.from_numpy(
                s.astype(np.float32)).reshape(1, 1, k_sobel, k_sobel, k_sobel)
            sobel_filter = sobel_filter.to(device, dtype=torch.float32)
            self.sobel_filters.append(sobel_filter)

    def __call__(self, img, a=5000, thresh=[0.2, 0.9]):
        '''
        Detect edges using Sobel operator for a 3d image.

        Parameters
        ----------
        img : torch tensor
            3D torch tensor of shape (b, c, x, y, z).
        a : int, optional
            padding to be added, do not change unless necessary. The default is 1.
        thresh : list of floats
            lower and upper bounds(exclusive) for filtering edges between (0,1). The default is [0.01,0.1].

        Returns
        -------
        torch tensor
            tensor of shape (b, 1, x, y, z) containing filtered edges.

        '''
        pad = (a, a, a, a, a, a)
        B, C, H, W, D = img.shape

        img = nn.functional.pad(img, pad, mode='reflect')

        grad_mag = (1 / C) * torch.sum(torch.stack([torch.sum(torch.cat([s(img[:, c:c+1])for c in range(
            C)], dim=1) + EPSILON, dim=1) ** 2 for s in self.sobel_filters], dim=1) + EPSILON, dim=1) ** 0.5
        grad_mag = grad_mag[:, a:-a, a:-a, a:-a]
        edges = norm(grad_mag.view(B, 1, H, W, D))

        lower_mask = torch.where(edges > thresh[0], 1, 0)
        upper_mask = torch.where(edges < thresh[1], 1, 0)
        edges = lower_mask * upper_mask

        return edges.detach().to(dtype=torch.float)


class NCCLoss(nn.Module):
    '''
    Simple implementation for Normalized Cross Correlation that can be
    minimized with upper-bound of alpha and lower-bound of 0.
    '''

    def __init__(self, alpha=100, grad_edges=True, device='cpu'):
        super(NCCLoss, self).__init__()
        self.NCC = None  # -1(very dissimilar) to 1(very similar)
        self.alpha = alpha

    def forward(self, y, yp):
        y_ = y - torch.mean(y)
        yp_ = yp - torch.mean(yp)
        self.NCC = torch.sum(
            y_ * yp_) / (((torch.sum(y_**2)) * torch.sum(yp_**2) + EPSILON)**0.5)

        error = (1 - self.NCC) * self.alpha

        return error


class SSDLoss(nn.Module):
    '''
    Simple implementation for Sum of Square Differance Loss.
    '''

    def __init__(self, alpha=3):
        super(SSDLoss, self).__init__()
        self.SSD = None
        self.alpha = alpha

    def forward(self, y, yp):
        self.SSD = torch.sum((y - yp)**2)
        error = self.SSD * self.alpha
        return error


class NMILoss(nn.Module):
    '''
    Simple implementation for Mutual Image Information Loss.
    '''

    def __init__(self, alpha=1000, bins=256, patch_size=100, bandwidth=3):
        super(NMILoss, self).__init__()
        self.bins = bins
        self.alpha = alpha
        self.patch = patch_size
        self.bandwidth = bandwidth

    def forward(self, y, yp):
        re_shape = self.patch * 2

        if len(y.shape) == 5:
            y = F.interpolate(
                y, size=(re_shape, re_shape, re_shape), mode='nearest')
            y = y.view(8 * y.shape[0] * y.shape[1],
                       self.patch, self.patch, self.patch)
            yp = F.interpolate(
                yp, size=(re_shape, re_shape, re_shape), mode='nearest')
            yp = yp.view(8 * yp.shape[0] * yp.shape[1],
                         self.patch, self.patch, self.patch)

        else:
            y = F.interpolate(y, size=(re_shape, re_shape), mode='nearest')
            y = y.view(4 * y.shape[0] * y.shape[1], self.patch, self.patch)
            yp = F.interpolate(yp, size=(re_shape, re_shape), mode='nearest')
            yp = yp.view(4 * yp.shape[0] *
                         yp.shape[1], self.patch, self.patch)

        nmi, _ = NMI(y, yp, self.bins, self.bandwidth)

        error = torch.mean(torch.abs(nmi - 1.) * self.alpha)
        return error


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
        self.tanh = torch.tanh

    def forward(self, x, max_translate=0.25):
        if len(x) > 3:
            psi, theta, phi = x[0], x[1], x[2]
            output = torch.stack((self.cos(psi) * self.cos(theta),
                                  self.sin(
                                      phi) * self.sin(psi) * self.cos(theta) - self.cos(phi) * self.sin(theta),
                                  self.cos(
                                      phi) * self.sin(psi) * self.cos(theta) + self.sin(phi) * self.sin(theta),
                                  max_translate * self.tanh(x[3]),
                                  self.cos(psi) * self.sin(theta),
                                  self.sin(phi) * self.sin(psi) * self.sin(theta) +
                                  self.cos(phi) * self.cos(theta),
                                  self.cos(phi) * self.sin(psi) * self.sin(theta) -
                                  self.sin(phi) * self.cos(theta),
                                  max_translate * self.tanh(x[4]),
                                  - self.sin(psi),
                                  self.sin(phi) * self.cos(psi),
                                  self.cos(phi) * self.cos(psi),
                                  max_translate * self.tanh(x[5]))).flatten()
        else:
            theta = x[0]
            output = torch.stack((self.cos(theta), - self.sin(theta),
                                 x[1], self.sin(theta), self.cos(theta), x[2])).flatten()
        return output


class Regressor(nn.Module):
    def __init__(self, moving, device):
        super(Regressor, self).__init__()
        if len(moving.shape) == 5:
            self.reg = nn.Parameter(torch.rand(
                (6), device=device), requires_grad=True)
        else:
            self.reg = nn.Parameter(torch.rand(
                (3), device=device), requires_grad=True)
        self.thetas = Theta()

    def forward(self):
        var = self.reg
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
