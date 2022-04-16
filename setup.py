from setuptools import setup, find_packages


setup(
      name='radtorch',
      version='1.0.0',
      version_date='04.15.2022',
      description='RADTorch, The Medical Imaging Machine Learning Framework',
      url='https://www.radtorch.com',
      author='Mohamed Elbanan, MD',
      author_email = "https://www.linkedin.com/in/mohamedelbanan/",
      license='GNU Affero General Public License v3.0 License',
      packages=find_packages(),
      install_requires=[
                        'numpy',
                        'matplotlib',
                        'pandas',
                        'pydicom' ,
                        'SimpleITK' ,
                        'torchinfo' ,
                        'shap',
                        'grad_cam',
                        'torch',
                        'torchvision',
                        'seaborn',
                        'albumentations',
                        'pillow',
                        'sklearn',
                        'tqdm',
                        'pathlib',
                        'seaborn-image'
                        ],


      zip_safe=True,
      classifiers=[
      "License :: OSI Approved :: GNU Affero General Public License v3.0 License",
      "Natural Language :: English",
      "Programming Language :: Python :: 3 :: Only",
      "Topic :: Software Development :: Libraries :: Python Modules",
      ]
      )
