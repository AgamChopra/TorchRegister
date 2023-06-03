# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 2023

@author: Agam Chopra
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from matplotlib import pyplot as plt

from utils import Regressor, Attention_UNet, NCCLoss, Edge3D, NMILoss


# Warping Function #
def get_affine_warp(theta, moving):
    if len(theta.shape) == 2:
        if theta.shape[-1] == 6:
            theta = theta.view(1, 2, 3)
        else:
            theta = theta.view(1, 3, 4)
    grid = F.affine_grid(theta, moving.size(), align_corners=False)
    warped = F.grid_sample(moving, grid, align_corners=False, mode='bilinear')
    return warped


# Affine Registration #
def affine_register(moving, target, lr=1E-5, epochs=1000, per=0.1, device='cpu', debug=True, criterions=None, weights=[0.33, 0.33, 0.33], grad_edges=True):
    if grad_edges:
        edge_filter = Edge3D(device=device)
        moving = edge_filter(moving)
        target = edge_filter(target)

    if criterions is None:
        criterions = [nn.MSELoss(), NCCLoss(device=device), NMILoss()]
    else:
        criterions = [nn.MSELoss()]
        weights = [1.]

    if len(moving.shape) == 5:
        regressor = nn.Sequential(nn.Linear(int(2 * per * (torch.flatten(
            moving).shape[0])), 64, bias=False), nn.ReLU(), nn.Linear(64, 12)).to(device=device)
        regressor[0].weight.data.zero_()
        regressor[2].weight.data.zero_()
        regressor[2].bias.data.copy_(torch.tensor(
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))
    else:
        regressor = nn.Sequential(nn.Linear(int(2 * per * (torch.flatten(
            moving).shape[0])), 32, bias=False), nn.ReLU(), nn.Linear(32, 6)).to(device=device)
        regressor[0].weight.data.zero_()
        regressor[2].weight.data.zero_()
        regressor[2].bias.data.copy_(torch.tensor(
            [1, 0, 0, 0, 1, 0], dtype=torch.float))

    params = regressor.parameters()
    optimizer = torch.optim.SGD(params, lr)
    regressor.train()
    losses_train = []

    idx = random.sample(range(0, torch.flatten(
        moving).shape[-1]), int(per * torch.flatten(moving).shape[0]))
    input_ = torch.cat((torch.flatten(moving).view(
        1, -1)[:, idx], torch.flatten(target).view(1, -1)[:, idx]), dim=1)

    for eps in trange(epochs):
        optimizer.zero_grad()

        theta = regressor(input_)  # 3D Affine Matrix
        if len(moving.shape) == 5:
            theta = theta.view(1, 3, 4)
        else:
            theta = theta.view(1, 2, 3)

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

    regressor.eval()
    final_theta = regressor(input_)
    if len(moving.shape) == 5:
        final_theta = final_theta.view(1, 3, 4)
    else:
        final_theta = final_theta.view(1, 2, 3)

    final_warped = get_affine_warp(final_theta, moving)

    return [final_warped, best_warped], [final_theta, best_theta]


# Rigid Registration #
def rigid_register(moving, target, lr=1E-5, epochs=1000, per=0.1, device='cpu', debug=True, criterions=None, weights=[0.33, 0.33, 0.33], grad_edges=True):
    if grad_edges:
        edge_filter = Edge3D(device=device)
        moving = edge_filter(moving)
        target = edge_filter(target)

    if criterions is None:
        criterions = [nn.MSELoss(), NCCLoss(device=device), NMILoss()]
    else:
        criterions = [nn.MSELoss()]
        weights = [1.]

    regressor = Regressor(moving, device)
    params = regressor.parameters()
    optimizer = torch.optim.SGD(params, lr)
    regressor.train()
    losses_train = []

    if debug:
        print(regressor.reg)

    for eps in trange(epochs):
        optimizer.zero_grad()

        theta = regressor()
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
                print(regressor.reg)

    final_theta = regressor()
    final_warped = get_affine_warp(final_theta, moving)

    return [final_warped, best_warped], [final_theta, best_theta]


# Flow Registeration #
class flow_register(nn.Module):
    def __init__(self, img_size, mode='bilinear', in_c=1, n=1, criterions=[nn.MSELoss(), NCCLoss(device='cuda'), NMILoss()], weights=[0.33, 0.33, 0.33], lr=1E-3, max_epochs=2000, stop_crit=1E-4):
        super(flow_register, self).__init__()
        self.model = Attention_UNet(
            img_size, mode, in_c=in_c, n=n)

        self.flow = None

        self.warp = None

        self.criterions, self.weights, self.lr, self.max_epochs, self.stop_crit = criterions, weights, lr, max_epochs, stop_crit

        params = self.model.parameters()

        self.optimizer = torch.optim.SGD(params, lr)

    def forward(self, x, device):
        y, self.flow = self.model(x, device)
        return y

    def optimize(self, moving, target, device, debug=True, grad_edges=False):
        if grad_edges:
            edge_filter = Edge3D(device=device)
            moving = edge_filter(moving)
            target = edge_filter(target)

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

            if losses_train[-1] <= self.stop_crit:
                message = 'Converged to %f' % self.stop_crit
                break

        if debug:
            print('Optimization ended with status: %s' % message)

    def deform(self, x):
        self.warp.eval()
        y = self.warp(x, self.flow)

        return y
