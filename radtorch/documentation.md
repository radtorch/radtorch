
## RadTorch

RadTorch is a package of higher level functions and classes that I found very handy during my journey of machine learning in diagnostic imaging. The package includes different functions and classes built upon pytorch, matplotlib and sckikit-learn.


The purpose of this tool kit is to provide users with a number of functions and classes that will save a lot of time which would have been otherwise spent converting DICOM images to other formats and modifying them to fit into modern machine learning and deep learning frameworks.

![](radtorch_stack.png)


## Who created RadTorch?
My name is Mohamed Elbanan. I am a Radiology Resident at Yale New Haven Health System, Clinical Research Associate at Yale School of Medicine and a Machine-learning enthusiast.

## How to install RadTorch?

RadTorch tool kit and its dependencies can be installed by downloading the tool kit source from the github repo [here](https://github.com/radtorch/radtorch)

then using the command line/terminal, navigate to the RadTorch tool kit folder and type:

```
pip3 install .
```

To uninstall simply use:

```
pip3 uninstall radtorch
```

## RadTorch Sub-Modules
RadTorch includes 5 sub-modules, each specialized in part of the machine learning pipeline:

### **radtorch.vis**
Contains functions/classes related to data visualization.

### **radtorch.data**
Contains functions/classes related to data import and preprocessing.

### **radtorch.dicom**
Contains functions/classes related to handling DICOM image objects.

### **radtorch.models**
Contains functions/classes related to creating and training machine learning models

### **radtorch.general**
Contains general purpose functions that do not belong to any sub-module


## Contributing
RadTorch is on GitHub. Bug reports and pull requests are welcome.


## License
**MIT License, Copyright (c) 2020 Mohamed Elbanan, MD**


Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
