# RPN2T
Robust Tracking Using Region Proposal Networks <br>
https://arxiv.org/pdf/1705.10447.pdf

### Introduction
RPN2T tracker achieved state-of-the-art results
on several large scale benchmarks including OTB50, OTB100 and VOT2016.

Detailed description of the system is provided by our paper(https://arxiv.org/pdf/1705.10447.pdf).

This software is implemented using [Caffe](https://github.com/BVLC/caffe/) and part of [R-CNN](https://github.com/ShaoqingRen/faster_rcnn).

### Citation

If you're using this code in a publication, please cite our paper.

    @article{  
      Jimmy2017RPN2T,  
      title={Robust Tracking Using Region Proposal Networks},  
      author={Ren, Jimmy and Yu, Zhiyang and Liu, Jianbo and Zhang, Rui and Sun, Wenxiu and Pang, Jiahao and Chen, Xiaohao and Yan, Qiong},  
      journal={arXiv preprint arXiv:1705.10447},  
      year={2017}  
    }

### System Requirements

This code is tested on 64 bit Linux (Ubuntu 14.04 LTS).

**Prerequisites**
  0. MATLAB (tested with R2014b)  
  0. Caffe (included in this repository `external/_caffe/`)   
  0. For GPU support, a GPU, CUDA toolkit and cuDNN will be needed. We have tested in `GTX TitanX(MAXWELL)` with `CUDA7.5+cuDNNv5` and `GTX 1080` with `CUDA8.0+cuDNNv5.1`.

### Installation

  > Compile Caffe according to the [installation guideline](http://caffe.berkeleyvision.org/installation.html).  
  ```shell  
  cd $(RPN2T_ROOT)
  cd external/_caffe
  # Adjust Makefile.config (For example, the path of MATLAB.)
  make all -j8
  make matcaffe
  ```  

### Online Tracking using RPN2T

**Demo**
  > Run 'tracking/demo_tracking.m'.
