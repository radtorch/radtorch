![](https://img.shields.io/badge/stable%20version-0.1.3_beta-blue)
![](https://img.shields.io/badge/nightly%20version-0.1.4_beta-yellow)
![](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen)
![](https://img.shields.io/badge/license-AGPL3.0-red)

# RADTorch
## The Radiology Machine Learning Framework

Official repository for RADTorch - The Radiology Machine Learning Framework


<p style='text-align: justify;'>
RADTorch provides a package of higher level functions and classes that significantly decrease the amount of time needed for implementation of different machine and deep learning algorithms on DICOM medical images.
</p>

![](/docs/img/radtorch_stack.png)


## How

Running a state-of-the-art DICOM image classifier can be run using the Image Classification Pipeline using the commands:
```
from radtorch import pipeline

classifier = pipeline.Image_Classification(data_directory='path to data')
classifier.run()
```
<small>
The above 3 lines of code will run an image classifier using VGG16 with pre-trained weights.
</small>

## See it in action
1. [Using RADTorch for classification of Normal vs COVID-19 Chest X-Rays.](https://www.kaggle.com/elbanan/radtorch-covid-19)
2. [Using RADTorch for classifying different animal images.](https://www.kaggle.com/elbanan/radtorch-animal-classification)



## Try it yourself
RADTorch playground for testing is provided on [Google Colab](https://colab.research.google.com/drive/1O7op_RtuNs12uIs0QVbwoeZdtbyQ4Q9i).


## Documentation
Full Documentation at official website: https://docs.radtorch.com


## Requests
Feature requests are more than welcomed on our discussion board [HERE](https://github.com/radtorch/radtorch/issues/4#issue-573590182)


## Who are we?
<p style='text-align: justify;'>
RADTorch was developed by Mohamed Elbanan, MD: a Radiology Resident at Yale New Haven Health System, Clinical Research Affiliate at Yale School of Medicine and a Machine-learning enthusiast.
</p>

#### Develpoment Team

1. [Mohamed Elbanan, MD](https://github.com/elbanan)
2. [Kareem Elfatairy, MD](https://github.com/kareemelfatairy)


## Contribute
Bug reports and pull requests are welcome.

## Cite us
This framework is provided for free and opensource to anyone who would like to use it and/or update it. 

Please cite us as :

```
@software{mohamed_elbanan_md_2020_3732781,
  author       = {Mohamed Elbanan, MD and Kareem Elfatairy, MD},
  title        = {RADTorch - The Radiology Machine Learning Tool Kit},
  month        = mar,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {v0.1.3b},
  doi          = {10.5281/zenodo.3732781},
  url          = {https://doi.org/10.5281/zenodo.3732781}
}
```
