from setuptools import setup

from radtorch.settings import version

setup(
      name='radtorch',
      version=version,
      description='RADTorch, The Radiology Machine Learning Tool Kit',
      url='https://radtorch.github.io/radtorch/',
      author='Mohamed Elbanan, MD',
      author_email = "https://www.linkedin.com/in/mohamedelbanan/",
      license='MIT',
      packages=['radtorch'],
      install_requires=['torch', 'torchvision', 'numpy', 'pandas', 'pydicom', 'matplotlib', 'pillow', 'tqdm', 'sklearn', 'efficientnet-pytorch','pathlib', 'bokeh'],
      zip_safe=False,
      classifiers=[
      "Development Status :: 4 - Beta",
      "License :: OSI Approved :: MIT License",
      "Natural Language :: English",
      "Programming Language :: Python :: 3 :: Only",
      "Topic :: Software Development :: Libraries :: Python Modules",
      ]
      )
