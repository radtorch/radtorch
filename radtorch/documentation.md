

RADTorch provides a package of higher level functions and classes that significantly decrease the coding time needed for implementation of different machine and deep learning algorithms on DICOM medical images. The purpose of this tool kit is to provide users with a number of functions and classes that will save a lot of time which would have been otherwise spent converting DICOM images to other formats and modifying them to fit into modern machine learning and deep learning frameworks.

RADTorch includes different functions and classes built upon pytorch, pydicom, matplotlib and sckikit-learn as seen in the stack chart below.




![](radtorch_stack.png)


## Who created RADTorch?
My name is Mohamed Elbanan. I am a Radiology Resident at Yale New Haven Health System, Clinical Research Affiliate at Yale School of Medicine and a Machine-learning enthusiast.

## How to install RADTorch?

RADTorch tool kit and its dependencies can be installed using the following terminal commands:

```
git clone https://github.com/radtorch/radtorch.git
pip3 install radtorch/.
```

To uninstall simply use:

```
pip3 uninstall radtorch
```

## RadTorch Sub-Modules
RADTorch includes 5 sub-modules, each specialized in part of the machine learning pipeline:

### **radtorch.pipeline**
The most exciting feature of RADTorch tool kit. Contains full machine learning pipelines that can be executed through a single line of code.

### **radtorch.visutils**
Contains functions/classes related to data visualization.

### **radtorch.datautils**
Contains functions/classes related to data import and preprocessing.

### **radtorch.dicomutils**
Contains functions/classes related to handling DICOM image objects.

### **radtorch.modelsutils**
Contains functions/classes related to creating and training machine learning models

### **radtorch.generalutils**
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
