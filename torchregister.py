# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 2023

@author: Agam Chopra
"""
import warpings as wrp


class Register():
    def __init__(self, mode='rigid', device='cpu', debug=False):
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

        Returns
        -------
        None.

        '''
        self.mode = mode
        self.warp = None if mode == 'flow' else wrp.get_affine_warp
        self.device = device
        self.debug = debug
        self.theta = None

    def optim(self, moving, target, lr=1E-5, max_epochs=1000, n=16, per=0.1):
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
            flowreg = wrp.flow_register(
                target.shape[2:], mode='bilinear', n=n, lr=lr, max_epochs=max_epochs).to(self.device)
            flowreg.optimize(moving, target, self.device, self.debug)
            self.theta = flowreg.flow
            self.warp = flowreg.deform

        elif self.mode == 'affine':
            _, theta = wrp.affine_register(
                moving, target, lr=lr, epochs=max_epochs, per=per, device=self.device, debug=self.debug)
            self.theta = theta[-1]
        else:
            _, theta = wrp.rigid_register(
                moving, target, lr=lr, epochs=max_epochs, per=per, device=self.device, debug=self.debug)
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
