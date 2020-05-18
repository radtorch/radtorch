![](welcome.png?raw=true)


![](https://img.shields.io/badge/stable%20version-1.1.1-green)
![](https://img.shields.io/badge/nightly%20version-1.1.2-lightgrey)
![](https://zenodo.org/badge/DOI/10.5281/zenodo.3827986.svg)
![](https://img.shields.io/badge/dependencies-up%20to%20date-green)
![](https://img.shields.io/badge/license-AGPL3.0-red)


<p style='text-align: justify;'>
<br>

RADTorch provides a framework of higher level classes and functions that aim at significantly reducing the time needed for implementation of different machine and deep learning algorithms on DICOM medical images.

RADTorch was **built by radiologists for radiologists** so they can build, test and implement state-of-the-art machine learning algorithms in minutes.


RADTorch is built upon widely used machine learning and deep learning frameworks. These include:

1. PyTorch for Deep Learning and Neural Networks.

2. Scikit-learn for Data Management and Machine Learning Algorithms.

3. PyDICOM for handling of DICOM data.

4. Bokeh, Matplotlib and Seaborn for Data Visualization.

</p>

<br>

### How

Running a state-of-the-art DICOM image classifier can be run using the Image Classification Pipeline using the commands:
```
from radtorch import pipeline

classifier = pipeline.Image_Classification(data_directory='path to data')
classifier.run()
```
<small>
The above 3 lines of code will run an image classifier using Alexnet model architecture with pre-trained weights for feature extraction and 'ridge' classifier.
</small>


<br>

### Try it yourself
RADTorch playground for testing is provided on [Google Colab](https://colab.research.google.com/drive/1O7op_RtuNs12uIs0QVbwoeZdtbyQ4Q9i).

<br>

### Documentation
Full Documentation at official website: https://docs.radtorch.com

<br>

### Requests
Feature requests are more than welcomed on our discussion board [HERE](https://github.com/radtorch/radtorch/issues/4#issue-573590182)

<br>

### Who are we?
<p style='text-align: justify;'>
RADTorch was developed by Mohamed Elbanan, MD: a Radiology Resident at Yale New Haven Health System, Clinical Research Affiliate at Yale School of Medicine and a Machine-learning enthusiast.
</p>

<br>

### Contribute
Bug reports and pull requests are welcome.

<br>

### Cite us
This framework is provided for free and opensource to anyone who would like to use it and/or update it.

Please cite us as :

```
@software{mohamed_elbanan_2020_3827986,
  author       = {Mohamed Elbanan},
  title        = {radtorch/radtorch: 1.1.1},
  month        = may,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {1.1.1},
  doi          = {10.5281/zenodo.3827986},
  url          = {https://doi.org/10.5281/zenodo.3827986}
}
```
