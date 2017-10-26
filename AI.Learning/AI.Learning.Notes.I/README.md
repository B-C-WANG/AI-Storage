# AI.Learning.CodeNotes
对现成的代码进行注释理解。
## dcgan_mnist.py
from https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/GAN/dcgan_mnist.py
###  discriminator, self.D: 鉴别器
-  **input**:  28 x 28 x 1, depth = 1 
-  **output**: 10 x 10 x 1, depth = 64
-  **structure**:


 layer | output_shape  
:--------:|:---------:
input | (None, 28, 28, 1)
**conv2d** | (None, 14, 14, 64)  
 dropout |- 
**conv2d** |   (None, 7, 7, 128)
dropout | -
**cond2d** | (None, 4, 4, 256)
dropout | -
**conv2d** | (None, 4, 4, 512)
dropout | -
flatten | (None, 8192)
dense | (None, 1)
activation| -

### generator, self.G: 生成器
- **input**:   100
- **output**:  28 x 28 x 1
- **structure**:

layer | output_shape
:----:|:-----:
input | (None,100)
dense | (None, 12544)
batch_normallization | -
activation|-
reshape | (None, 7, 7, 256)
dropout | -
upSampling | (None, 14, 14, 256)
**deconv2d** | (None, 14, 14, 128)
batch_normalization | -
activation |-
upSampling| (None, 28, 28, 128)
**deconv2d** | (None, 28, 28, 64)
batch_normalization | -
activation | -
**deconv2d** | (None, 28, 28, 32)
batch_normalization |-
activation |-
deconv2d | (None, 28, 28, 1)
activation | -


### discriminator_model, self.DM, 鉴别模型
- **based on self.D**
- **optimizer**: RMSprop
- **loss**: binary_crossentropy

### adversarial_model, self.AM 对抗模型:
- **based on self.G and self.D**
- **optimizer**: RMSprop
- **loss**: binary_crossentropy

#### **鉴别**
1. 随机获取batch_size大小原始MNIST的image_train数据，shape=[batch_size, 100]
2. 获取batch_size大小的噪声数据(随机数范围-1.0~1.0)，shape=[batch_size, 100]
3. 用生成模型将噪声数据转变为image数据，得到image_fake
4. 组合噪声和原始数据，各占一半
5. 生成label数据，前一半全为1（对应原始minst数据），后一半全为0（对应噪声生成的假数据）。
6. **用DM进行训练（即训练权重，使其能够将原始和生成的假数据分开，这是鉴别模型**）

#### **对抗**
1. 生成全1的label数据（对应原始真实图片）
2. 生成噪声数据[batch_size, 100]
3. **用AM模型训练，先用noise生成假图片数据，然后再鉴别，调整权重（此时生成和鉴别模型的权重是统一训练的，选用一个优化器），使鉴别器最终能够得到1。**
4. **注意这里的鉴别器和上面的鉴别器是一样的，或权重相同的，如果不同，则模型独立开来，达不到对抗的目的，由于上面的鉴别器致力于训练权重分开真假数据，而下面的鉴别器又致力于训练权重使随机数据能够得到和真实数据相同的标签（标签1），因而相互对抗，互相增强。最终使生成的数据能够接近真实数据**

#### **可视化**
灌入随机数据到生成模型中，生成最终的图片

#### **总结思路**
将真实图像数据和随机数经过生成器生成的虚假数据选用不同标签训练鉴别器。
将生成器和鉴别器连在一起，训练使其将随机数生成的虚假图像数据能够辨别为1.
#### **遇到的问题**
1. 为什么要将生成器和鉴别器分开，直接训练生成器使其能够生成鉴别器认为是1的图片不就行了，这样训练可能也会使鉴别器中的权重受影响————直接生成虚假图片后很难看出其真实性，所以需要加上鉴别器鉴别，才能计算误差梯度传播，可能只改变生成器的权重要更好一些。
2. 生成器的意义————它实际上就是一个生成图片数据的算法，你可以用其他生成的数据算法代替，但是神经网络在此是很好用的，它需要一个100维度的随机数数据，然后用权重将其转化为虚假图片数据，这里面的权重（也包括偏差），就是重要的信息。你之后生成图片就只需要给100个随机数就可以了。（更高级的算法可以给语义，然后生成图片）

EOF:)
