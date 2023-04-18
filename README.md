<h1>TorchRegister</h1>
<p>Common medical 3D image registration methods such as rigid, affine, and flow field for PyTorch.</p>

<p align="center">
    <img width="900" height="200" src="https://github.com/AgamChopra/TorchGradRegister/blob/main/assets/flow_test.jpg">
    <br><i>Fig. Example visualizations of deep learning based flow-field brain MRI registration.</i><br><br>
    <img width="900" height="240" src="https://github.com/AgamChopra/TorchGradRegister/blob/main/assets/affine_test.jpg">
    <br><i>Fig. Example visualizations of PyTorch based Affine brain MRI registration.</i><br><br>
    <img width="900" height="240" src="https://github.com/AgamChopra/TorchGradRegister/blob/main/assets/rigid_test.jpg">
    <br><i>Fig. Example visualizations of PyTorch based Rigid brain MRI registration.</i><br><br>
    <img width="200" height="150" src="https://github.com/AgamChopra/TorchGradRegister/blob/main/assets/flow_test_loss.png"> 
    <img width="200" height="150" src="https://github.com/AgamChopra/TorchGradRegister/blob/main/assets/affine_test_loss.png">
    <img width="200" height="150" src="https://github.com/AgamChopra/TorchGradRegister/blob/main/assets/rigid_test_loss.png">
    <br><i>Fig. Example visualizations of loss curve for flow-field registration(left), affine registration(middle), and rigid registration(right).</i><br><br>
</p>

<p>Example:</p>
 <pre><code>
 
        import torch
        from torchio.transforms import RandomAffine
        from numpy import load
        from matplotlib import pyplot as plt
        from torchregister import Register


        # augmentation function
        def rand_augment(x):
            affine = RandomAffine(image_interpolation='bspline',
                                  degrees=45, translation=8, scales=(0.7, 1.5))
            y = affine(x[0])
            return y.view(x.shape)


        device = 'cuda'

        # loading data
        path = 'example_mri.pkl'
        data = load(path, allow_pickle=True)
        
        moving = torch.from_numpy(data)
        moving = moving.view(1, 1, moving.shape[0], moving.shape[1], moving.shape[2]).to(
            dtype=torch.float, device=device)
        
        target = torch.from_numpy(data)
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
</code></pre>

<p><a href="https://raw.githubusercontent.com/AgamChopra/TorchGradRegister/main/LICENSE.md" target="blank">[The MIT License]</a></p>
