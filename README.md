<h1>TorchRegister</h1>
<p>Common 2D and 3D image registration methods such as rigid, affine, and flow field for PyTorch.</p>

<p align="center">
    <img width="900" height="240" src="https://github.com/AgamChopra/TorchGradRegister/blob/main/assets/ringo_registration_2d.jpg">
    <br><i>Fig. Example visualizations 2D image registration.</i><br><br>
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
        import TorchRegister as tr


        def rand_augment(x):
            affine = RandomAffine(image_interpolation='bspline',
                                  degrees=25, translation=4, scales=(0.8, 1.2))
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

        plt.imshow(torch.squeeze(moving[:, :, :, :, 60]
                                 ).detach().cpu().numpy(), cmap='gray')
        plt.title('Moving')
        plt.show()

        plt.imshow(torch.squeeze(target[:, :, :, :, 60]
                                 ).detach().cpu().numpy(), cmap='gray')
        plt.title('Target')
        plt.show()

        # Rigid registration
        warping = tr.Register(mode='rigid', device=device, debug=False)
        warping.optim(moving, target, max_epochs=500, lr=1E-5)
        warped = warping(moving)

        plt.imshow(torch.squeeze(warped[:, :, :, :, 60]
                                 ).detach().cpu().numpy(), cmap='gray')
        plt.title('Warped Moving 1')
        plt.show()

        # Affine registration
        moving = warped.detach()
        warping = tr.Register(mode='affine', device=device, debug=False)
        warping.optim(moving, target, max_epochs=200, lr=1E-5)
        warped = warping(moving)

        plt.imshow(torch.squeeze(warped[:, :, :, :, 60]
                                 ).detach().cpu().numpy(), cmap='gray')
        plt.title('Warped Moving 2')
        plt.show()

        # Flow field based registration
        moving = warped.detach()
        warping = tr.Register(mode='flow', device=device, debug=False)
        warping.optim(moving, target, lr=1E-3, max_epochs=100)
        warped = warping(moving)

        plt.imshow(torch.squeeze(warped[:, :, :, :, 60]
                                 ).detach().cpu().numpy(), cmap='gray')
        plt.title('Warped Moving 3')
        plt.show()

        plt.imshow(torch.moveaxis(torch.squeeze(tr.norm(
            torch.abs(warping.theta[:, :, :, :, 60]))), 0, -1).detach().cpu().numpy())
        plt.title('Flow Field')
        plt.show()
</code></pre>

<p><a href="https://raw.githubusercontent.com/AgamChopra/TorchGradRegister/main/LICENSE.md" target="blank">[The MIT License]</a></p>
