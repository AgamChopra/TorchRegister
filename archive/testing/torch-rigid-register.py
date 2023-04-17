# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 2023

@author: Agam Chopra
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchio.transforms import RandomAffine
from numpy import load
from matplotlib import pyplot as plt
from tqdm import trange
import random


class Theta(nn.Module):
    def __init__(self):
        super(Theta, self).__init__()
        self.activation = F.tanh

    def forward(self, x):
        output = x.clone()
        output[:, 0] = self.activation(output[:, 0])
        output[:, 1] = 2 * self.activation(output[:, 1])
        output[:, 2] = 2 * self.activation(output[:, 2])

        output[:, 4] = self.activation(output[:, 4])
        output[:, 5] = 2 * self.activation(output[:, 5])
        output[:, 6] = 2 * self.activation(output[:, 6])

        output[:, 8] = self.activation(output[:, 8])
        output[:, 9] = self.activation(output[:, 9])
        output[:, 10] = self.activation(output[:, 10])
        return output


class Regressor(nn.Module):
    def __init__(self, moving, per, device):
        super(Regressor, self).__init__()
        self.reg = nn.Sequential(nn.Linear(int(2 * per * (torch.flatten(
            moving).shape[0])), 64, bias=False), nn.ReLU(), nn.Linear(64, 12)).to(device=device)
        self.reg[0].weight.data.zero_()
        self.reg[2].weight.data.zero_()
        self.reg[2].bias.data.copy_(torch.tensor(
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))
        self.thetas = Theta()

    def forward(self, input_):
        var = self.reg(input_)
        theta = self.thetas(var)
        return theta.view(1, 3, 4)


def get_affine_warp(theta, moving):
    grid = F.affine_grid(theta, moving.size(), align_corners=False)
    warped = F.grid_sample(moving, grid, align_corners=False, mode='bilinear')
    return warped


def rigid_register(moving, target, lr=1E-5, epochs=1000, per=0.1, device='cpu', debug=True):
    if debug:
        plt.imshow(torch.squeeze(
            moving[:, :, :, :, 60]).detach().cpu().numpy(), cmap='gray')
        plt.title('Moving')
        plt.show()

        plt.imshow(torch.squeeze(
            target[:, :, :, :, 60]).detach().cpu().numpy(), cmap='gray')
        plt.title('Target')
        plt.show()

    regressor = Regressor(moving, per, device)
    params = regressor.parameters()
    optimizer = torch.optim.SGD(params, lr)
    criterions = [nn.MSELoss(), nn.L1Loss()]
    weights = [0.5, 0.5]

    losses_train = []

    idx = random.sample(range(0, torch.flatten(
        moving).shape[-1]), int(per * torch.flatten(moving).shape[0]))
    input_ = torch.cat((torch.flatten(moving).view(
        1, -1)[:, idx], torch.flatten(target).view(1, -1)[:, idx]), dim=1)

    for eps in trange(epochs):
        optimizer.zero_grad()

        theta = regressor(input_)
        warped = get_affine_warp(theta, moving)

        error = sum([weights[i] * criterions[i](target, warped)
                    for i in range(len(criterions))])
        error.backward()
        optimizer.step()

        losses_train.append(error.item())

        if eps == 0:
            loss_low = error.item()
            best_warped = warped
            best_theta = theta
        else:
            if error.item() < loss_low:
                loss_low = error.item()
                best_warped = warped
                best_theta = theta

        if debug:
            if (eps % (epochs/10) == 0 or eps == epochs - 1) and eps != 0:
                plt.plot(losses_train, label='Error')
                plt.title('Optimization Criterion')
                plt.xlabel('Epoch')
                plt.ylabel('Error')
                plt.legend()
                plt.show()

                plt.imshow(torch.squeeze(
                    warped[:, :, :, :, 60]).detach().cpu().numpy(), cmap='gray')
                plt.title('Warped Moving')
                plt.show()

    final_theta = regressor(input_)
    final_warped = get_affine_warp(final_theta, moving)

    return [final_warped, best_warped], [final_theta, best_theta]


def test(device='cpu'):
    def rand_augment(x):
        affine = RandomAffine(image_interpolation='bspline',
                              degrees=25, translation=10)
        y = affine(x[0])
        return y.view(x.shape)

    path = 'R:/img (%d).pkl' % (1)
    data = load(path, allow_pickle=True)
    moving = torch.from_numpy(data[0])
    moving = moving.view(1, 1, moving.shape[0], moving.shape[1], moving.shape[2]).to(
        dtype=torch.float, device=device)
    target = torch.from_numpy(data[0])
    target = rand_augment(target.view(1, 1, target.shape[0], target.shape[1], target.shape[2])).to(
        dtype=torch.float, device=device)

    warps, _ = rigid_register(
        moving, target, lr=1E-5, epochs=1000, per=0.1, device=device, debug=True)

    print(target.shape)
    print(moving.shape)
    print(warps[0].shape)
    print(warps[1].shape)

    plt.imshow(torch.squeeze(moving[:, :, :, :, 60]
                             ).detach().cpu().numpy(), cmap='gray')
    plt.title('Moving')
    plt.show()

    plt.imshow(torch.squeeze(warps[0][:, :, :, :, 60]
                             ).detach().cpu().numpy(), cmap='gray')
    plt.title('Final Warped Moving')
    plt.show()

    plt.imshow(torch.squeeze(warps[1][:, :, :, :, 60]
                             ).detach().cpu().numpy(), cmap='gray')
    plt.title('Best Warped Moving')
    plt.show()

    plt.imshow(torch.squeeze(target[:, :, :, :, 60]
                             ).detach().cpu().numpy(), cmap='gray')
    plt.title('Target')
    plt.show()


if __name__ == '__main__':
    test('cuda')
