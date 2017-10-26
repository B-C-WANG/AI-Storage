# Play With Keras.js
**最近发现了一个keras网页部署工具keras.js，项目地址：https://github.com/transcranial/keras-js**
**项目给出了一些demo可供参考，网址：https://transcranial.github.io/keras-js/#/**
**这些demo对于理解神经网络非常有帮助，以往的理解是基于模型和一大堆矩阵数字，现在这个demo将其可视化出来，非常便于理解！**
## Basic Convnet for MNIST
![](http://i.imgur.com/1Dfhf7l.png)

以手写数字5为例，见图片Basic_Convnet_for_MNIST.png
1. 一开始经过conv2d卷积后，由于卷积核对于不同的形貌敏感程度不同，图像的对比度和明暗程度有了变化，进一步activation，然后再次conv2d卷积后，逐渐出现了数字的特征
2. 在maxpooling2D中可以看到，数字5除了在前面颜色较深，后面也有一点儿，推测是数字8的区域，其卷积核与5有些许类似
3. 也就可以看出，卷积核的权重是随空间分布的，从左至右，卷积核分别提取0,1,2...9的特征，其中有些就是0,1特征的混合，有些是1,2特征的混合

## Convolutional Variational Autoencoder
![](http://i.imgur.com/E6375v6.png)
见图片Convolutional_Variational_Autoencoder.png
1. 这个例子是卷积变分自编码器的decoder部分
2. 在latent space中点击不同区域，会生成不同的手写数字，相近区域的数字也是相似的，比如9,7和1就是相邻的
3. VAE将特征提取之后，在latent space中进行了区分

## AC-GAN
![](http://i.imgur.com/uth3Qix.png)
见图片acGAN.png
**这个例子只列出了GAN的生成器，GAN使用100个随机噪声数据来产生机器生成的手写图像。相比于VAE，VAE只在latent sapce用两个点生成图像。GAN生成的图像更丰富，而VAE旨在保留原始特征**

## Bidirectional LSTM 
![](http://i.imgur.com/6SgGNDq.png)
见图Bidirectional_LSTM.png
**用双向神经网络实现语句的情绪识别**
