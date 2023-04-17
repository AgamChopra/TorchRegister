# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 2023

@author: Agam Chopra
"""
import torch
from torchio.transforms import RandomAffine
from numpy import load
from matplotlib import pyplot as plt

from torchregister import Register


def rand_augment(x):
    affine = RandomAffine(image_interpolation='bspline',
                          degrees=45, translation=8, scales=(0.7, 1.5))
    y = affine(x[0])
    return y.view(x.shape)


device = 'cuda'

# loading data
path = 'R:/img (%d).pkl' % (1)
data = load(path, allow_pickle=True)
moving = torch.from_numpy(data[0])
moving = moving.view(1, 1, moving.shape[0], moving.shape[1], moving.shape[2]).to(
    dtype=torch.float, device=device)
target = torch.from_numpy(data[0])
target = rand_augment(target.view(1, 1, target.shape[0], target.shape[1], target.shape[2])).to(
    dtype=torch.float, device=device)

# Flow field based registration
warping = Register(mode='flow', device=device, debug=True)
warping.optim(moving, target, lr=1E-3)
warped = warping(moving)

print(target.shape)
print(moving.shape)
print(warped[0].shape)

plt.imshow(torch.squeeze(moving[:, :, :, :, 60]
                         ).detach().cpu().numpy(), cmap='gray')
plt.title('Moving')
plt.show()

plt.imshow(torch.squeeze(warped[:, :, :, :, 60]
                         ).detach().cpu().numpy(), cmap='gray')
plt.title('Warped Moving')
plt.show()

plt.imshow(torch.squeeze(target[:, :, :, :, 60]
                         ).detach().cpu().numpy(), cmap='gray')
plt.title('Target')
plt.show()

# Affine registration
warping = Register(mode='affine', device=device, debug=True)
warping.optim(moving, target)
warped = warping(moving)

print(target.shape)
print(moving.shape)
print(warped[0].shape)

plt.imshow(torch.squeeze(moving[:, :, :, :, 60]
                         ).detach().cpu().numpy(), cmap='gray')
plt.title('Moving')
plt.show()

plt.imshow(torch.squeeze(warped[:, :, :, :, 60]
                         ).detach().cpu().numpy(), cmap='gray')
plt.title('Warped Moving')
plt.show()

plt.imshow(torch.squeeze(target[:, :, :, :, 60]
                         ).detach().cpu().numpy(), cmap='gray')
plt.title('Target')
plt.show()

# Rigid registration
warping = Register(mode='rigid', device=device, debug=True)
warping.optim(moving, target)
warped = warping(moving)

print(target.shape)
print(moving.shape)
print(warped[0].shape)

plt.imshow(torch.squeeze(moving[:, :, :, :, 60]
                         ).detach().cpu().numpy(), cmap='gray')
plt.title('Moving')
plt.show()

plt.imshow(torch.squeeze(warped[:, :, :, :, 60]
                         ).detach().cpu().numpy(), cmap='gray')
plt.title('Warped Moving')
plt.show()

plt.imshow(torch.squeeze(target[:, :, :, :, 60]
                         ).detach().cpu().numpy(), cmap='gray')
plt.title('Target')
plt.show()
