# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 17:46:42 2023

@author: Agam Chopra
"""
import torch
import torch.nn as nn
from math import ceil
from numpy import min, max
from pytorch_msssim import SSIM
from matplotlib import pyplot as plt
from tqdm import trange


class PSNR():
    def __init__(self, epsilon=1E-9):
        self.name = "PSNR"
        self.epsilon = epsilon

    def __call__(self, x, y):
        mse = torch.mean((x - y) ** 2)
        psnr = 20 * torch.log10(torch.max(x)) - 10 * torch.log10(mse)
        loss = (psnr + self.epsilon) ** -1
        return loss


class ssim_loss(nn.Module):
    def __init__(self, channel=1, spatial_dims=3, win_size=11, win_sigma=1.5):
        super(ssim_loss, self).__init__()
        self.ssim = SSIM(channel=channel, spatial_dims=spatial_dims,
                         win_size=win_size, win_sigma=win_sigma)

    def forward(self, x, y):
        loss = 1 - self.ssim(x, y)
        return loss


def norm(x):
    EPSILON = 1E-9
    try:
        return (x - torch.min(x)) / ((torch.max(x) - torch.min(x)) + EPSILON)
    except torch.is_tensor(x):
        try:
            return (x - min(x)) / ((max(x) - min(x)) + EPSILON)
        except Exception:
            print('WARNING: Input could not be normalized!')


def pad3d(input_, target, device='cpu'):
    delta = [target.shape[2+i] - input_.shape[2+i] for i in range(3)]
    return nn.functional.pad(input=input_, pad=(ceil(delta[2]/2), delta[2] - ceil(delta[2]/2),
                                                ceil(
                                                    delta[1]/2), delta[1] - ceil(delta[1]/2),
                                                ceil(delta[0]/2), delta[0] - ceil(delta[0]/2)),
                             mode='constant', value=0).to(dtype=torch.float, device=device)


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
    def __init__(self, x_c, g_c, i_c, stride=3, mode='nearest'):
        super(attention_grid, self).__init__()
        self.input_filter = nn.Conv3d(
            in_channels=x_c, out_channels=i_c, kernel_size=1, stride=stride, bias=False)
        self.gate_filter = nn.Conv3d(
            in_channels=g_c, out_channels=i_c, kernel_size=1, stride=1, bias=True)
        self.psi = nn.Conv3d(in_channels=i_c, out_channels=1,
                             kernel_size=1, stride=1, bias=True)
        self.bnorm = nn.BatchNorm3d(i_c)
        self.mode = mode

    def forward(self, x, g, device):
        x_shape = x.shape

        a = self.input_filter(x)
        b = self.gate_filter(g)

        if a.shape[-1] < b.shape[-1]:
            a = pad3d(a, b, device)

        elif a.shape[-1] > b.shape[-1]:
            b = pad3d(b, a, device)

        w = torch.sigmoid(self.psi(nn.functional.relu(a + b)))
        w = nn.functional.interpolate(w, size=x_shape[2:], mode=self.mode)

        y = x * w
        y = self.bnorm(y)
        return y, w


class grad_register(nn.Module):
    '''
    Non-linear model for 3D image registration via overfitting.
    '''

    def __init__(self, img_size, mode='nearest', in_c=1, out_c=3, n=1, criterions=[nn.MSELoss(), nn.L1Loss(), ssim_loss(win_size=3, win_sigma=0.1), PSNR()], weights=[0.15, 0.15, 0.60, 0.20], lr=1E-3, max_epochs=2000, stop_crit=1E-5):
        super(grad_register, self).__init__()
        self.down = nn.Sequential(nn.Conv3d(in_channels=in_c, out_channels=int(64/n), kernel_size=3), nn.ReLU(), nn.BatchNorm3d(int(64/n)),
                                  nn.Conv3d(in_channels=int(
                                      64/n), out_channels=int(64/n), kernel_size=3), nn.ReLU(), nn.BatchNorm3d(int(64/n)),
                                  nn.Conv3d(in_channels=int(64/n), out_channels=int(64/n), kernel_size=2, stride=2), nn.ReLU(), nn.BatchNorm3d(int(64/n)))

        self.skip = attention_grid(int(64/n), int(64/n), int(64/n))

        self.latent = nn.Sequential(nn.Conv3d(in_channels=int(64/n), out_channels=int(128/n), kernel_size=3), nn.ReLU(), nn.BatchNorm3d(int(128/n)),
                                    nn.Conv3d(in_channels=int(
                                        128/n), out_channels=int(128/n), kernel_size=3), nn.ReLU(), nn.BatchNorm3d(int(128/n)),
                                    nn.ConvTranspose3d(in_channels=int(128/n), out_channels=int(64/n), kernel_size=2, stride=2), nn.ReLU(), nn.BatchNorm3d(int(64/n)))

        self.up = nn.Sequential(nn.Conv3d(in_channels=int(128/n), out_channels=int(64/n), kernel_size=3), nn.ReLU(), nn.BatchNorm3d(int(64/n)),
                                nn.Conv3d(in_channels=int(64/n), out_channels=int(64/n), kernel_size=3), nn.ReLU(), nn.BatchNorm3d(int(64/n)))

        self.out = nn.Conv3d(in_channels=int(
            64/n), out_channels=out_c, kernel_size=1)

        self.warp = SpatialTransformer(img_size, mode)

        self.flow = None

        self.criterions, self.weights, self.lr, self.max_epochs, self.stop_crit = criterions, weights, lr, max_epochs, stop_crit

        params = list(self.down.parameters()) + list(self.latent.parameters()) + list(self.skip.parameters()
                                                                                      ) + list(self.up.parameters()) + list(self.out.parameters()) + list(self.warp.parameters())

        self.optimizer = torch.optim.Adam(params, self.lr)

    def forward(self, x, device):
        y1 = self.down(x)

        y2 = self.latent(y1)
        y1, _ = self.skip(y1, y2, device=device)

        y = torch.cat((y1, pad3d(y2, y1, device=device)), dim=1)
        y = self.up(y)

        y = pad3d(y, x, device=device)

        self.flow = self.out(y)

        y = self.warp(x, self.flow)

        return y

    def optimize(self, moving, target, device, debug=True):
        losses_train = []
        message = 'Reached max epochs'
        self.train()

        for eps in trange(self.max_epochs):
            self.optimizer.zero_grad()

            y = self(moving, device)

            error = sum([self.weights[i] * self.criterions[i](target, y)
                        for i in range(len(self.criterions))])
            error.backward()
            self.optimizer.step()

            losses_train.append(error.item())

            if debug:
                if (eps % 100 == 0 or eps == self.max_epochs - 1) and eps != 0:
                    plt.plot(losses_train, label='Error')
                    plt.title('Optimization Criterion')
                    plt.xlabel('Epoch')
                    plt.ylabel('Error')
                    plt.legend()
                    plt.show()

            if losses_train[-1] <= self.stop_crit:
                message = 'Converged to %f' % self.stop_crit
                break

        if debug:
            print('Optimization ended with status: %s' % message)

    def register(self, x):
        y = self.warp(x, self.flow)

        return y


def test(device='cpu'):
    a = torch.zeros((1, 1, 80, 80, 80), device=device)
    a[:, :, 10:20, 10:20,
        10:20] += torch.ones((1, 1, 10, 10, 10), device=device)
    b = torch.zeros((1, 1, 80, 80, 80), device=device)
    b[:, :, 50:60, 50:60,
        50:60] += torch.ones((1, 1, 10, 10, 10), device=device)
    c = torch.zeros((1, 1, 80, 80, 80), device=device)
    c[:, :, 10:20, 10:20,
        10:20] += torch.rand((1, 1, 10, 10, 10), device=device)

    model = grad_register(a.shape[2:], mode='bilinear').to(device)
    model.optimize(b, a, device)
    d = model.register(c)

    print(a.shape)
    print(b.shape)
    print(c.shape)
    print(d.shape)


if __name__ == '__main__':
    test('cuda')
