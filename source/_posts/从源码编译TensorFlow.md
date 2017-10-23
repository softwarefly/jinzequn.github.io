---
title: 从源码编译TensorFlow
date: 2017-04-18 12:44:21
type: "categories"
tags: TensorFlow
categories: TensorFlow
toc: True

---

最近接到一个深度学习用于关键点定位的项目，开始应用深度学习进行图像的识别。考察了现有的几个深度学习框架，无疑在图像领域__Caffe__还是最专业的一个深度学习框架。但是Caffe的入门比较困难，后来看到Github上TensorFlow的项目比较多。时间紧，任务重，好入门就是王道。所以毅然决然的踏入了TensorFlow的坑。这一系列的文章将会贯穿我整个项目过程，打上TensorFlow的标签进行记录。本文配置基于Ubuntu16.04LTS，Windows用户出门左转。
<!--more-->
## 1、安装依赖库
TensorFlow的安装需要依赖库，包括一些必要的运行环境。装上都用的到，也不占多大地方，强迫症自行解决。

    sudo apt-get install -y build-essential git python-pip libfreetype6-dev libxft-dev libncurses-dev libopenblas-dev gfortran python-matplotlib libblas-dev liblapack-dev libatlas-base-dev python-dev python-pydot linux-headers-generic linux-image-extra-virtual unzip python-numpy swig python-pandas python-sklearn unzip wget pkg-config zip g++ zlib1g-dev

    sudo pip install -U pip
## 2、配置CUDA和Cudnn
这两个库的配置参考该官方网站，配置过程简单不在赘述。

http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#axzz4VZnqTJ2A

https://developer.nvidia.com/cudnn 

从官方网站下载好cuda和cudnn以后用以下命令安装。

    dpkt -i cuda文件名
    sudo apt update
    sudo apt install cuda

cudnn配置的过程简而言之就是把下载好的cudnn复制到相应的cuda库中，下载速度会有限制，自行科学上网。因为要复制到`/usr/local/cuda-8.0`中，所以要用命令行给root权限。


__注意！ 接下来需要配置环境变量。打开一个终端。__
   
    cd
    gedit .bashrc
    
然后将如下代码复制粘贴在末尾。

    export CUDA_HOME=/usr/local/cuda-8.0
    export CUDA_ROOT=/usr/local/cuda-8.0
    export PATH=$PATH:$CUDA_ROOT/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_ROOT/lib64
    
然后使环境变量生效。

    source .bashrc
## 3、安装JAVA8
    sudo add-apt-repository ppa:webupd8team/java
    sudo apt-get update
    sudo apt-get install oracle-java8-installer
安装过程有图形化界面，在许可证问题上选Yes，可以通过Tab键进行选择。
## 4、安装bazel
    echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
    curl https://storage.googleapis.com/bazel-apt/doc/apt-key.pub.gpg | sudo apt-key add -
    sudo apt-get update && sudo apt-get install bazel
    sudo apt-get upgrade bazel
这一步可能需要科学上网，否则速度会很慢从而导致失败
## 5、源码编译TensorFlow
在合适的位置打开终端，从github上下载Tensorflow仓库。

     git clone https://github.com/tensorflow/tensorflow

完成之后，在打开其文件夹，然后开始进行配置。

    cd tensorflow
    ./configure
**注意：**

1、只需要将选项 build tensorflow with cuda support 之后写y，其他选项都直接回车，默认不支持。

2、CUDA SDK version 写 8.0。

3、Cudnn version 写5。

4、Compute capability 写 6.1。（这里的值需要自己查询电脑GPU型号对应的计算能力，我的电脑是GTX1080，所以对应6.1）

## 6、生成pip安装包

    bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
    bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
    
从第二行命令可以看出来，生成的.whl安装包将会放置在/tmp/tensorflow_pkg

## 7、安装pip安装包

    sudo pip install /tmp/tensorflow_pkg/tensorflow-1.0.1-py2-none-any.whl

.whl的名字是不固定的，安装的时候要看 /tmp/tensorflow_pkg/文件夹下具体是什么名字。至此安装过程全部完成。

## 8、多说一段
本次的安装过程是进行GPU版的TensorFlow的安装。这样就不再用在程序中写代码指定GPu了，直接会启用GPU。非GPU版的安装直接用pip安装即可

    sudo pip install tensorflow
    
TensorFlow是一个强大的开源深度学习框架，就目前做的工作来看，深度学习对于浅层网络还是有绝对的优势。但是深度学习依赖于大数据，人工标注数据成本很高，不过会有各种新的模型出现以解决这个问题吧。

