# BSD-GAN
Tensorflow implementation of BSD-GAN


[![BSD-GAN demo](https://i.ytimg.com/vi/syXtdQFz_yY/1.jpg?time=1521770504792)](https://www.youtube.com/watch?v=syXtdQFz_yY&t=34s)

 # BSD-GAN paper
<a href="https://arxiv.org/abs/1803.08467">paper</a>

please cite the paper, if the codes/dataset has been used for your research.

# results of BSD-GAN

![Scale-aware fusion](https://github.com/duxingren14/BSD-GAN/blob/master/teaser.png)

# How to setup

## Prerequisites

* Python (2.7 or later)

* numpy

* scipy

* NVIDIA GPU + CUDA 8.0 + CuDNN v5.1

* TensorFlow 1.0 or later


# Getting Started
## steps

* clone this repo:

```
git clone https://github.com/duxingren14/BSD-GAN.git

cd BSD-GAN

mkdir data
```

* download datasets (e.g., car) from <a href="https://drive.google.com/open?id=1cLvx0qECkgAV5rZGsrS30FtVOOOGkeim">google drive</a> and put it in ./data/.

./BSD-GAN/data/car_400x300/\*.jpg 


* train the model:

```
python main.py --phase train --dataset_name car_400x300 --input_height 300 --input_width 400 --epoch 45
```

* test the model:

```
python main.py --phase train --dataset_name car_400x300 --input_height 300 --input_width 400 --epoch 45 --train False

```

# other datasets

some datasets can be downloaded from the website. Please cite their papers if you use the data. 

celebahq: https://github.com/tkarras/progressive_growing_of_gans

lsun: http://lsun.cs.princeton.edu/2017/


# Experimental results:

![celeba_hq256](https://github.com/duxingren14/BSD-GAN/blob/master/face256.png)
![celeba_hq512](https://github.com/duxingren14/BSD-GAN/blob/master/face512.png)
![car_400x300](https://github.com/duxingren14/BSD-GAN/blob/master/car.png)
![lsun church_outdoor](https://github.com/duxingren14/BSD-GAN/blob/master/lsun.png)

# Acknowledgments

Codes are built on the top of <a href="https://github.com/tkarras/progressive_growing_of_gans">DCGAN tensorflow</a> and <a href="https://github.com/tkarras/progressive_growing_of_gans">progressive growing</a>. Thanks for their precedent contributions!

