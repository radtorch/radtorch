![](https://img.shields.io/badge/stable%20version-0.1.2_beta-blue)
![](https://img.shields.io/badge/nightly%20version-0.1.3_beta-yellow)
![](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen)
![](https://img.shields.io/badge/license-AGPL3.0-red)

# RADTorch - The Radiology Machine Learning Tool Kit

Official repository for RADTorch - The Radiology Machine Learning Tool Kit



## About
<p style='text-align: justify;'>
RADTorch provides a package of higher level functions and classes that significantly decrease the amount of time needed for implementation of different machine and deep learning algorithms on DICOM medical images.
</p>

<p style='text-align: justify;'>
RADTorch was developed and is currently maintained by Mohamed Elbanan, MD: a Radiology Resident at Yale New Haven Health System, Clinical Research Affiliate at Yale School of Medicine and a Machine-learning enthusiast.
</p>

![](/docs/img/radtorch_stack.png)



## Getting Started

Running a state-of-the-art DICOM image classifier can be run using the Image Classification Pipeline using the commands:
```
from radtorch import pipeline

classifier = pipeline.Image_Classification(data_directory='path to data')
classifier.run()
```
<small>
The above 3 lines of code will run an image classifier using VGG16 with pre-trained weights.
</small>


## Documentation
Full Documentation: https://docs.radtorch.com

## Playground
RADTorch playground for testing is provided on [Google Colab](https://colab.research.google.com/drive/1O7op_RtuNs12uIs0QVbwoeZdtbyQ4Q9i).


## Feature Requests

Feature requests are more than welcomed on our discussion board [HERE](https://github.com/radtorch/radtorch/issues/4#issue-573590182)

## Contributing
Bug reports and pull requests are welcome.
