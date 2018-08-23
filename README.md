# The code and pre-trained model will be released in a few days!

# Pixel2Mesh
This repository contains the TensorFlow implementation for the following paper</br>

[Pixel2Mesh: Generating 3D Mesh Models from Single RGB Images (ECCV2018)](https://arxiv.org/abs/1804.01654)</br>

Nanyang Wang*, [Yinda Zhang](http://robots.princeton.edu/people/yindaz/)\*, [Zhuwen Li](http://www.lizhuwen.com/)\*, [Yanwei Fu](http://yanweifu.github.io/), [Wei Liu](http://www.ee.columbia.edu/~wliu/), [Yu-Gang Jiang](http://www.yugangjiang.info/). (*Equal Contribution)

The code is based on the [gcn](https://github.com/tkipf/gcn) framework. For Chamfer losses, we have included the cuda implementations of [Fan et. al](https://github.com/fanhqme/PointSetGeneration).

# Project Page
The project page is available at http://bigvid.fudan.edu.cn/pixel2mesh

# Dependencies
Requirements:
* Python2.7+ with Numpy and opencv-python
* [Tensorflow (version 1.0+)](https://www.tensorflow.org/install/)
* [TFLearn](http://tflearn.org/installation/)

Our code has been tested with Python 2.7, **TensorFlow 1.3.0**, TFLearn 0.3.2, CUDA 8.0 on Ubuntu 14.04.

# Installation
    git clone https://github.com/nywang16/Pixel2Mesh.git
    cd Pixel2Mesh
    python setup.py install
    
# Running the demo
    python test_nn.py data/examples/plane.png
Run the testing code and the output mesh file is saved in data/examples/plane.obj

**Input image, output mesh.**</br>
<img src="./pictures/plane.png" width = "330px" />
![label](./pictures/plane.gif)

# Dataset

We used the [ShapeNet](https://www.shapenet.org) dataset for 3D models, and rendered views from [3D-R2N2](https://github.com/chrischoy/3D-R2N2):</br>
When using the provided data make sure to respect the shapenet [license](https://shapenet.org/terms).

Below is the complete set of training data. Download it into the data/ folder.
https://drive.google.com/file/d/1Z8gt4HdPujBNFABYrthhau9VZW10WWYe/view?usp=sharing </br>

    cd pixel2mesh/data
    tar -xzf ShapeNetTrain.tar


The training/testing split can be found in data/`train_list.txt` and data/`test_list.txt` </br>
    
The file is named in syntheticID_modelID_renderID.dat format, and the data processing script will be released soon.

Each .dat file in the provided data contain: </br>
* The rendered image from 3D-R2N2. We resized it to 224x224 and made the background white.
* The sampled point cloud (with vertex normal) from ShapeNet. We transformed it to corresponding coordinates in camera coordinate based on camera parameters from the Rendering Dataset.

**Input image, ground truth point cloud.**</br>
<img src="./pictures/car_example.png" width = "350px" />
![label](./pictures/car_example.gif)

# Training
    python train_nn.py
You can change the training data, learning rate and other parameters by editing `train_nn.py`

# Citation
If you use this code for your research, please consider citing:

    @inProceedings{wang2018pixel2mesh,
      title={Pixel2Mesh: Generating 3D Mesh Models from Single RGB Images},
      author={Nanyang Wang and Yinda Zhang and Zhuwen Li and Yanwei Fu and Wei Liu and Yu-Gang Jiang},
      booktitle={ECCV},
      year={2018}
    }
