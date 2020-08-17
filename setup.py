
from setuptools import setup, find_packages


setup(
      name='radtorch',
      version='0.08.16',
      version_date='08.16.2020',
      description='RADTorch, The Radiology Machine Learning Framework',
      url='https://www.radtorch.com',
      author='Mohamed Elbanan, MD',
      author_email = "https://www.linkedin.com/in/mohamedelbanan/",
      license='GNU Affero General Public License v3.0 License',
      packages=find_packages(),
      install_requires=['tornado==5.1.1', 'torch', 'torchvision', 'torchsummary', 'numpy', 'pandas', 'pydicom', 'matplotlib', 'pillow',
                        'tqdm', 'sklearn','pathlib', 'bokeh', 'xgboost', 'seaborn', 'torchsummary', 'efficientnet_pytorch',
                        'xmltodict',
                        # 'streamlit',
#                         'detectron2 @ git+https://github.com/facebookresearch/detectron2.git#egg=v0.1.3',
#                         'pyyaml>=5.1'
                       ],

      zip_safe=False,
      classifiers=[
      "License :: OSI Approved :: GNU Affero General Public License v3.0 License",
      "Natural Language :: English",
      "Programming Language :: Python :: 3 :: Only",
      "Topic :: Software Development :: Libraries :: Python Modules",
      ]
      )
