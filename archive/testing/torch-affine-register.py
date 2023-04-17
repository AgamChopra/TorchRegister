# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 2023

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


def get_affine_warp(theta, moving):
    grid = F.affine_grid(theta, moving.size(), align_corners=False)
    warped = F.grid_sample(moving, grid, align_corners=False, mode='bilinear')
    return warped


def affine_register(moving, target, lr=1E-5, epochs=1000, per=0.1, device='cpu', debug=True):
    if debug:
        plt.imshow(torch.squeeze(
            moving[:, :, :, :, 60]).detach().cpu().numpy(), cmap='gray')
        plt.title('Moving')
        plt.show()

        plt.imshow(torch.squeeze(
            target[:, :, :, :, 60]).detach().cpu().numpy(), cmap='gray')
        plt.title('Target')
        plt.show()

    regressor = nn.Sequential(nn.Linear(int(2 * per * (torch.flatten(
        moving).shape[0])), 64, bias=False), nn.ReLU(), nn.Linear(64, 12)).to(device=device)
    regressor[0].weight.data.zero_()
    regressor[2].weight.data.zero_()
    regressor[2].bias.data.copy_(torch.tensor(
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))
    params = regressor.parameters()
    optimizer = torch.optim.SGD(params, lr)
    criterions = [nn.MSELoss(), nn.L1Loss()]
    weights = [0.5, 0.5]

    regressor.train()
    losses_train = []

    idx = random.sample(range(0, torch.flatten(
        moving).shape[-1]), int(per * torch.flatten(moving).shape[0]))
    input_ = torch.cat((torch.flatten(moving).view(
        1, -1)[:, idx], torch.flatten(target).view(1, -1)[:, idx]), dim=1)

    for eps in trange(epochs):
        optimizer.zero_grad()

        theta = regressor(input_).view(1, 3, 4)  # 3D Affine Matrix
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

    regressor.eval()
    final_theta = regressor(input_).view(1, 3, 4)
    final_warped = get_affine_warp(final_theta, moving)

    return [final_warped, best_warped], [final_theta, best_theta]


def test(device='cpu'):
    def rand_augment(x):
        affine = RandomAffine(image_interpolation='bspline',
                              degrees=45, translation=8, scales=(0.7, 1.5))
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

    warps, _ = affine_register(
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
