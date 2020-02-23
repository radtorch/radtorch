

## About RADTorch

<p style='text-align: justify;'>
RADTorch provides a package of higher level functions and classes that significantly decrease the amount of time needed for implementation of different machine and deep learning algorithms on DICOM medical images. The most important feature of RADTorch tool kit is probably "piplines" which, for example, allows users to obtain a state of the art medical image classifier with as little as 2 lines of code. The different functions and classes included in RADTorch are built upon PyTorch, PyDICOM, Matplotlib and Scikit-learn.
</p>

<p style='text-align: justify;'>
RADTorch was developed and is currently maintained by Mohamed Elbanan, MD: a Radiology Resident at Yale New Haven Health System, Clinical Research Affiliate at Yale School of Medicine and a Machine-learning enthusiast.
</p>

![](radtorch_stack.png)

<br>
## Install RADTorch

RADTorch tool kit and its dependencies can be installed using the following terminal commands:

```
git clone github.com/radtorch/radtorch.git
pip3 install radtorch/.
```

To uninstall simply use:

```
pip3 uninstall radtorch
```


<br>
## Quick Start Guide
Running a state of the art DICOM image classifier can be run using the [Image Classification](./pipeline.html#radtorch.pipeline.Image_Classification) Pipeline using the commands:
```
from radtorch import pipeline

classifier = pipeline.Image_Classification(data_directory='path to data directory')
classifier.train()
```
<small>
The above 3 lines of code will run an image classifier using VGG16 with pre-trained weights.
</small>


<br>
## Supported Neural Network Architectures

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;border-color:#ccc;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:7px 16px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-top-width:1px;border-bottom-width:1px;border-color:#ccc;color:#333;background-color:#fff;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:7px 16px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-top-width:1px;border-bottom-width:1px;border-color:#ccc;color:#333;background-color:#f0f0f0;}
.tg .tg-m5nv{border-color:#656565;text-align:center;vertical-align:top}
.tg .tg-hkgo{font-weight:bold;border-color:#656565;text-align:left;vertical-align:top}
.tg .tg-dfrc{background-color:#f9f9f9;border-color:#656565;text-align:left;vertical-align:top}
.tg .tg-09jq{background-color:#f9f9f9;border-color:#656565;text-align:center;vertical-align:top}
.tg .tg-2bev{border-color:#656565;text-align:left;vertical-align:top}
</style>
<table class="tg">
  <tr>
    <th class="tg-hkgo">Model Architecture     </th>
    <th class="tg-hkgo">Default Image Size</th>
    <th class="tg-hkgo">Number of Output Features</th>
  </tr>
  <tr>
    <td class="tg-dfrc">VGG16</td>
    <td class="tg-09jq">224x224</td>
    <td class="tg-09jq">4096</td>
  </tr>
  <tr>
    <td class="tg-2bev">VGG18</td>
    <td class="tg-m5nv">224x224</td>
    <td class="tg-m5nv">4096</td>
  </tr>
  <tr>
    <td class="tg-dfrc">ResNet50</td>
    <td class="tg-09jq">224x224</td>
    <td class="tg-09jq">2048</td>
  </tr>
  <tr>
    <td class="tg-2bev">ResNet101</td>
    <td class="tg-m5nv">224x224</td>
    <td class="tg-m5nv">2048</td>
  </tr>
  <tr>
    <td class="tg-dfrc">ResNet152</td>
    <td class="tg-09jq">224x224</td>
    <td class="tg-09jq">2048</td>
  </tr>
    <tr>
    <td class="tg-dfrc">wide_resnet50_2</td>
    <td class="tg-09jq">224x224</td>
    <td class="tg-09jq">2048</td>
  </tr>
  <tr>
    <td class="tg-dfrc">wide_resnet101_2</td>
    <td class="tg-09jq">224x224</td>
    <td class="tg-09jq">2048</td>
  </tr>
  <tr>
    <td class="tg-dfrc">Inception v3</td>
    <td class="tg-09jq">299x299</td>
    <td class="tg-09jq">2048</td>
  </tr>     
</table>


<br>
## Contributing to RADTorch
RadTorch is on [GitHub](https://github.com/radtorch/radtorch). Bug reports and pull requests are welcome.


<br>
## License
MIT License, Copyright 	&copy; 2020 Mohamed Elbanan, MD

<p style='text-align: justify;'>
<small>
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
</small>
</p>

<p style='text-align: justify;'>
<small>
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
</small>
</p>
