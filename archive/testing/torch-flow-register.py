# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 2023

@author: Agam Chopra
"""
import torch
import torch.nn as nn
from torchio.transforms import RandomAffine
from math import ceil
from numpy import min, max, moveaxis, load
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


# !!! NEEDS REWORK. Ref. to https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
# Should not use flow field for this type of registration
# 2 Step registration:
    # Ridgid: predict variables, apply ridgid deformation
    # Affine: predict variables, apply affine deformation
# Calculate Error and backprop
# Make a sperate flow registration method using current logic
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
        self.bnorm = nn.InstanceNorm3d(i_c)
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


class Attention_UNet(nn.Module):
    def __init__(self, img_size, mode='nearest', in_c=1, out_c=3, n=1):
        super(Attention_UNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv3d(in_channels=in_c, out_channels=int(64/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm3d(int(64/n)),
                                    nn.Conv3d(in_channels=int(64/n), out_channels=int(64/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm3d(int(64/n)))

        self.skip1 = attention_grid(int(64/n), int(64/n), int(64/n))

        self.layer2 = nn.Sequential(nn.Conv3d(in_channels=int(64/n), out_channels=int(128/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm3d(int(128/n)),
                                    nn.Conv3d(in_channels=int(128/n), out_channels=int(128/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm3d(int(128/n)))

        self.skip2 = attention_grid(int(128/n), int(128/n), int(128/n))

        self.layer3 = nn.Sequential(nn.Conv3d(in_channels=int(128/n), out_channels=int(256/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm3d(int(256/n)),
                                    nn.Conv3d(in_channels=int(256/n), out_channels=int(256/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm3d(int(256/n)))

        self.skip3 = attention_grid(int(256/n), int(256/n), int(256/n))

        self.layer4 = nn.Sequential(nn.Conv3d(in_channels=int(256/n), out_channels=int(512/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm3d(int(512/n)),
                                    nn.Conv3d(in_channels=int(512/n), out_channels=int(512/n), kernel_size=3), nn.ReLU(), nn.InstanceNorm3d(int(512/n)))

        self.skip4 = attention_grid(int(512/n), int(512/n), int(512/n))

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

        y = torch.cat((y4, pad3d(y, y4, device=device)), dim=1)
        y = self.layer6(y)
        y3, _ = self.skip3(y3, y, device=device)

        y = torch.cat((y3, pad3d(y, y3, device=device)), dim=1)
        y = self.layer7(y)
        y2, _ = self.skip2(y2, y, device=device)

        y = torch.cat((y2, pad3d(y, y2, device=device)), dim=1)
        y = self.layer8(y)
        y1, _ = self.skip1(y1, y, device=device)

        y = torch.cat((y1, pad3d(y, y1, device=device)), dim=1)
        y = self.layer9(y)

        y = pad3d(y, x, device=device)

        flow = self.out(y)

        y = self.warp(x, flow)

        return y, flow


class flow_register(nn.Module):
    '''
    Non-linear model for 3D image registration via overfitting.
    '''

    def __init__(self, img_size, mode='bilinear', in_c=1, out_c=3, n=1, criterions=[nn.MSELoss(), nn.L1Loss(), ssim_loss(win_size=3, win_sigma=0.1), PSNR()], weights=[0.15, 0.15, 0.60, 0.20], lr=1E-3, max_epochs=2000, stop_crit=1E-4):
        super(flow_register, self).__init__()
        self.model = Attention_UNet(
            img_size, mode, in_c=in_c, out_c=out_c, n=n)

        self.flow = None

        self.warp = None

        self.criterions, self.weights, self.lr, self.max_epochs, self.stop_crit = criterions, weights, lr, max_epochs, stop_crit

        params = self.model.parameters()

        self.optimizer = torch.optim.Adam(params, self.lr)

    def forward(self, x, device):
        y, self.flow = self.model(x, device)
        return y

    def optimize(self, moving, target, device, debug=True):
        losses_train = []
        message = 'Reached max epochs'
        self.train()

        if debug:
            plt.imshow(torch.squeeze(
                moving[:, :, :, :, 60]).detach().cpu().numpy(), cmap='gray')
            plt.title('Moving')
            plt.show()

            plt.imshow(torch.squeeze(
                target[:, :, :, :, 60]).detach().cpu().numpy(), cmap='gray')
            plt.title('Target')
            plt.show()

        for eps in trange(self.max_epochs):
            self.optimizer.zero_grad()

            y = self(moving, device)

            error = sum([self.weights[i] * self.criterions[i](target, y)
                        for i in range(len(self.criterions))])
            error.backward()
            self.optimizer.step()

            self.warp = self.model.warp

            losses_train.append(error.item())

            if debug:
                if (eps % 100 == 0 or eps == self.max_epochs - 1) and eps != 0:
                    plt.plot(losses_train, label='Error')
                    plt.title('Optimization Criterion')
                    plt.xlabel('Epoch')
                    plt.ylabel('Error')
                    plt.legend()
                    plt.show()

                    plt.imshow(torch.squeeze(
                        y[:, :, :, :, 60]).detach().cpu().numpy(), cmap='gray')
                    plt.title('Warped Moving')
                    plt.show()

                    plt.imshow(moveaxis(torch.squeeze(
                        norm(torch.abs(self.flow[:, :, :, :, 60]))).detach().cpu().numpy(), 0, -1))
                    plt.title('Flow Field')
                    plt.show()

            if losses_train[-1] <= self.stop_crit:
                message = 'Converged to %f' % self.stop_crit
                break

        if debug:
            print('Optimization ended with status: %s' % message)

    def deform(self, x):
        self.warp.eval()
        y = self.warp(x, self.flow)

        return y


def test(device='cpu'):
    def rand_augment(x):
        affine = RandomAffine(image_interpolation='bspline',
                              degrees=15, translation=5)
        y = affine(x[0])
        return y.view(x.shape)

    path = 'R:/img (%d).pkl' % (1)
    data = load(path, allow_pickle=True)

    moving = torch.from_numpy(data[0]).view(
        1, 1, data[0].shape[0], data[0].shape[1], data[0].shape[2]).to(device=device, dtype=torch.float)
    target = rand_augment(torch.from_numpy(data[0]).view(
        1, 1, data[0].shape[0], data[1].shape[1], data[0].shape[2])).to(device=device, dtype=torch.float)

    flowreg = flow_register(target.shape[2:], mode='bilinear', n=16).to(device)
    flowreg.optimize(moving, target, device)
    d = flowreg.deform(moving)

    plt.imshow(torch.squeeze(
        moving[:, :, :, :, 60]).detach().cpu().numpy(), cmap='gray')
    plt.title('Moving')
    plt.show()

    plt.imshow(torch.squeeze(
        d[:, :, :, :, 60]).detach().cpu().numpy(), cmap='gray')
    plt.title('Warped Moving')
    plt.show()

    plt.imshow(torch.squeeze(
        target[:, :, :, :, 60]).detach().cpu().numpy(), cmap='gray')
    plt.title('Target')
    plt.show()


if __name__ == '__main__':
    test('cuda')
