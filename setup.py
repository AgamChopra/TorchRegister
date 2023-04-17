from setuptools import setup, find_packages

setup(
    name='TorchRegister',
    version='0.1.0',
    url='https://github.com/AgamChopra/TorchRegister'
    author='Agam Chopra'
    author_email='achopra4@uw.edu'
    description='Common medical 3D image registration methods such as rigid, affine, and flow field for PyTorch.'
    packages=find_packages(),    
    install_requires=['torch', 'numpy', 'tqdm', 'matplotlib'],
)
