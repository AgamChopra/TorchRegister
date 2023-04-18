from setuptools import setup, find_packages

setup(
    name='TorchRegister',
    version='0.1.1',
    url='https://github.com/AgamChopra/TorchRegister',
    author='Agam Chopra',
    author_email='achopra4@uw.edu',
    description='3D image registration methods for PyTorch.',
    long_description='Common medical 3D image registration methods such as rigid, affine, and flow field for PyTorch.',
    install_requires=['torch >= 2.0.0', 'numpy >= 1.24.1',
                      'tqdm >= 4.65.0', 'matplotlib  >= 3.7.1'],
    package_dir={'': "src"},
    packages=find_packages("src"),
)
