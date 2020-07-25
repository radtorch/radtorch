![](welcome.png?raw=true)


![](https://img.shields.io/badge/stable%20version%20date-2020.08.01-green)
![](https://zenodo.org/badge/DOI/10.5281/zenodo.3827986.svg)
![](https://img.shields.io/badge/dependencies-up%20to%20date-green)
![](https://img.shields.io/badge/license-AGPL3.0-red)


### What is RADTorch 
<p style='text-align: justify;'>

RADTorch provides a framework of higher level classes and functions that aim at significantly reducing the time needed for implementation of different machine and deep learning algorithms on DICOM medical images.

RADTorch was **built by radiologists for radiologists** so they can build, test and implement state-of-the-art machine learning algorithms in minutes.


RADTorch is built upon widely used machine learning and deep learning frameworks. These include:

1. PyTorch for Deep Learning and Neural Networks.

2. Scikit-learn for Data Management and Machine Learning Algorithms.

3. PyDICOM for handling of DICOM data.

4. Bokeh, Matplotlib and Seaborn for Data Visualization.

</p>

<br>



### Update Log

    6/20/2020

    - Hybrid Clinical Features/Imaging features Image Classification pipeline.
    

    5/14/2020

    - CNN visualization with Class Activation Maps.

    - Deep Learning Model Summary.

    - WGAN added to GAN models.

    - EfficientNet models added to Image Classification


<br>


### Getting Started

Running a state-of-the-art DICOM image classifier can be run using the Image Classification Pipeline using the commands:
```
from radtorch import pipeline

classifier = pipeline.Image_Classification(data_directory='path to data')
classifier.run()
```
<small>
The above 3 lines of code will run an image classifier using vgg16 model architecture with pre-trained weights for feature extraction and 'logistic_regression' classifier.
</small>

<br>

<br>

### See it in Action

1. [Image Classification for identification of contrast on CT DICOM images](https://www.kaggle.com/elbanan/radtorch-ct-contrast-id)

2. [Image classification demo/non medical on Kaggle](https://www.kaggle.com/elbanan/radtorch-demo)

3. [RADTorch playground on Google Colab](https://colab.research.google.com/drive/1O7op_RtuNs12uIs0QVbwoeZdtbyQ4Q9i).

<br>

### Documentation
Full Documentation at official website: https://docs.radtorch.com

<br>

### Requests
Feature requests are more than welcomed on our discussion board [HERE](https://github.com/radtorch/radtorch/issues/4#issue-573590182)

<br>


### Contribute
Bug reports and pull requests are welcome.

<br>

### Contributors
<p style='text-align: justify;'>
RADTorch was developed by Mohamed Elbanan, MD: a Radiology Resident at Yale New Haven Health System, Clinical Research Affiliate at Yale School of Medicine and a Machine-learning enthusiast.
</p>

<br>

### Cite us
This framework is provided for free and opensource to anyone who would like to use it and/or update it.

Please cite us as :

```
@software{mohamed_elbanan_2020_3902428,
  author       = {Mohamed Elbanan},
  title        = {radtorch/radtorch: 2020.06.20},
  month        = jun,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {2020.06.20},
  doi          = {10.5281/zenodo.3902428},
  url          = {https://doi.org/10.5281/zenodo.3902428}
}
```
