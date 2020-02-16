from setuptools import setup

setup(name='radtorch',
      version='0.1',
      description='RadTorch, The Radiology Machine Learning Tool Kit',
      url='https://github.com/radtorch/radtorch',
      author='Mohamed Elbanan, MD',
      license='MIT',
      packages=['radtorch'],
      install_requires=['torch', 'torchvision', 'numpy', 'pandas', 'pydicom', 'matplotlib', 'pillow', 'tqdm', 'sklearn'],
      zip_safe=False)
