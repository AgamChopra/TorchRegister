# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 2023

@author: Agam Chopra
"""
from .warpings import flow_register, affine_register, rigid_register, get_affine_warp


class Register():
    def __init__(self, mode='rigid', device='cpu', debug=False, criterion=None, weights=None, optim='SGD'):
        '''
        Pytorch based numerical registration methods

        Parameters
        ----------
        mode : string, optional
            'rigid', 'affine', 'flow'. The default is 'rigid'.
        device : string, optional
            device to perform registration/optimization on. The default is 'cpu'.
        debug : string, optional
            print outputs of registration and loss curve every nth epoch. The default is False.
        criterion : list of nn.losses, optional
            criterion used to calculate registration error. The default is None ie- [nn.MSELoss(), nn.L1Loss()].
        weights : list of floats, optional
            weights associated with criterion. The default is None ie- [0.5, 0.5].
        optim : string, optional
            optimizer to use, SGD or ADAM. The default is 'SGD'.


        Returns
        -------
        None.

        '''
        self.criterion = criterion
        self.weights = weights
        self.optim = optim
        self.mode = mode
        self.warp = None if mode == 'flow' else get_affine_warp
        self.device = device
        self.debug = debug
        self.theta = None

    def optim(self, moving, target, lr=1E-5, max_epochs=1000, n=32, per=0.1):
        '''
        Optimization loop to get deformation matrix/flow-field

        Parameters
        ----------
        moving : tensor
            Tensor of shape [1,1,x,y,z] to be warped.
        target : tensor
            Target tensor of shape [1,1,x,y,z].
        lr : float, optional
            Learning rate. The default is 1E-5.
        max_epochs : int, optional
            Number of optimization iterations. The default is 1000.
        n : int, optional
            Parameter scaling factor for flow-field model. The default is 16.
        per : float, optional
            Percentage of voxels to be randomly sampled for registration. Value should be between 0 and 1. The default is 0.1.

        Returns
        -------
        None.

        '''
        if self.mode == 'flow':
            if self.criterion is not None and self.weights is not None:
                flowreg = flow_register(target.shape[2:], mode='bilinear', n=n, lr=lr, max_epochs=max_epochs, criterions=self.criterion, weights=self.weight, optim=self.optim).to(self.device)
            else:
                flowreg = flow_register(target.shape[2:], mode='bilinear', n=n, lr=lr, max_epochs=max_epochs, optim=self.optim).to(self.device)
            flowreg.optimize(moving, target, self.device, self.debug)
            self.theta = flowreg.flow
            self.warp = flowreg.deform

        elif self.mode == 'affine':
            if self.criterion is not None and self.weights is not None:
                _, theta = affine_register(moving, target, lr=lr, epochs=max_epochs, per=per, device=self.device, debug=self.debug, criterions=self.criterion, weights=self.weight, optim=self.optim)
            else:
                _, theta = affine_register(moving, target, lr=lr, epochs=max_epochs, per=per, device=self.device, debug=self.debug, optim=self.optim)         
            self.theta = theta[-1]
            
        else:
            if self.criterion is not None and self.weights is not None:
                _, theta = rigid_register(moving, target, lr=lr, epochs=max_epochs, per=per, device=self.device, debug=self.debug, criterions=self.criterion, weights=self.weight, optim=self.optim)
            else:
                _, theta = rigid_register(moving, target, lr=lr, epochs=max_epochs, per=per, device=self.device, debug=self.debug, optim=self.optim)  
            self.theta = theta[-1]

    def __call__(self, moving):
        '''
        Warp moving image using deformation obtained from optim

        Parameters
        ----------
        moving : tensor
            Tensor of shape [1,1,x,y,z] to be warped.

        Returns
        -------
        warped_moving : tensor
            Warped tensor of shape [1,1,x,y,z].

        '''
        if self.mode == 'flow':
            warped_moving = self.warp(moving)
        else:
            warped_moving = self.warp(self.theta, moving)
        return warped_moving
