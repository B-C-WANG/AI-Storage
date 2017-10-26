## addition_rnn.py
让机器学会加法运算
**如遇运行报错，检查tensorflow版本，多为tf.concat和tf.unstack的问题**
1.  首先创建字符词典，由于是加法，所以字符词典的键值为[0,1,2,3,4,5,6,7,8,9,+]
2.  encode函数，将算式转化为独热编码的矩阵，如35+35，即第一行第4个为1，第二行第6个为1，第三行最后一个为1......其余全为0，共5行，之后设置了最大长度为7行，之后将添加全0的两行凑成7行（用于两个三位数相加）
3.  decode函数，是encode的逆过程
4.  之后生成算式数据，随机生成两个位数随机的数字，encode成输入，计算出结果，encode成输出，输入拓展成7行，输出拓展成4行（适用于最高两个3位数相加）
5.  按10%比例划分训练和测试集
6.  训练模型，输出原始算式、正确结果和预测结果，模型如下表：


 layer | output_shape  
:--------:|:---------:
input | (None, 7, 12)
LSTM | (None, 128)
repeat_vector | (None, 4, 128)
LSTM | (None, 4, 128)
time_distributed | (None, 4, 12)
activation(softmax) | (None, 4, 12)
## antirectifier.py
创建一个自定义的层(layer)
1. 创建一个类，这个类继承自layers.Layer，即class custom_layer(layers.Layer)
2. 这个例子中重写了父类的compute_output_shape(self,input_shape)和call(self,inputs)函数，在连接层的时候，会调用前者去计算得到的shape，调用后者输出output。可以尝试查看layers.Layer的源代码，重写其他方法。
3. 此处compute_output_shape将input的第二维乘以2输出output_shape，而call则先L2正则化，然后输出正和负的ReLU输出结果，将其连接，于是得到2倍长度的矩阵，可以当做正则化和激活函数使用。

## mnist_acgan.py
**判别器模型：**

layer | output_shape
:----:|:-----:
input | (None, 1, 28, 28)
conv2d | (None, 32, 14, 14)
LeakyReLU | -
dropout | -
conv2d | (None, 64, 14, 14)
LeakyReLU | -
dropput | -
conv2d | （None, 128, 7, 7）
LeakyReLU | -
dropput | -
conv2d | (None, 256, 7, 7)
LeakyReLU | -
dropput | -
flatten | (None, 12544)
输出1: fake |输出2： auxiliary
**output1: fake** |Dense(1, activation="sigmoid") 
**output2: aux** |Dense(10, activation="softmax") 
- **判别器return:**
- Input(用于feed image数据): shape为(1, 28, 28)
- Output(用于feed label数据用以训练，或predict得到结果): [fake(shape=(None, 1)), aux(shape=(None, 10))]，fake是真假布尔值，aux是对应的手写数字预测的数字标签(one-hot)

**生成器模型：**
例子中的参数，latent_size = 100

layer | output_shape
:----:|:-----:
input | (None, 100)
dense | (None, 1024)
dense | (None, 128 x 7 x 7=6272)
reshape |(None, 128, 7, 7)
upsampling | (None, 128, 14, 14) 
conv2d | (None, 256, 14, 14)
upsampling | (None, 256, 28, 28)
conv2d | (None, 128, 28, 28)
cond2d | (None, 1, 28, 28)
- **生成器return：**
- Input(用于feed随机数数据): shape为(None, 100)的随机数，shape为(1，None)的手写数字标签(非one-hot)。
- Output(用于feed图片数据用以训练，或predict生成假图片数据): fake_image(shape=(None, 1, 28, 28))

### 过程

1. 设定参数，依照模型建立损失函数
2. 设置placeholder作为生成器的输入（100长度的随机数以及1个长度的label），得到生成后的假图片；之后将假图片作为输入进判别器中，得到fake和auxiliary
3. **combine生成和判别模型，合成一个模型，输入100长度随机数和1长度的label，输出是否为真的布尔值以及auxiliary**
4. 接着准备原始手写输入数据，用生成器基于100个(-1, 1)的浮点随机数以及1个(0, 10)的整数，生成假图片，然后混入原始图片中
5. 设置真假标签y，真实图片的fake=1，生成的图片fake=0,
6. 设置识别标签aux，真实图片aux为对应的label，生成的图片aux为对应生成图片的随机数
6. 将原始图片数据，随机数生成的假图片数据作为image，真假标签y作为fake，识别标签aux作为aux，一同喂入鉴别器中进行训练，此时是在训练鉴别器鉴别真假的能力
7. 之后将两随机数，以及**全为1的fake布尔值**和对应的标签，一同喂入3中所述组成的生成-判别模型，进行训练，为的就是训练生成模型，**使其生成的图片能够让判别模型得出的fake布尔值为1，并且生成的图片被判别器识别出的数字，和一开始输入的随机整数相同(也就是训练给定一个100位的随机数以及1个数字，生成这个数字的手写图像)**
8. 上述的GAN训练完成之后，混入真实数据进行测试，计算loss，并输出生成的图片进行显示


### 理解
- 我们可以这样理解这种监督学习的训练过程：给定输入输出，使网络自行调整以符合输入输出，比如输入真实图片的label为1，虚假图片label为0，就是训练辨别能力
- 而输入虚假图片，又使判别器输出为1，还要使判别器对于虚假图片对应的手写数字的判别与初始的随机数相同，这就是在训练使其能够达成以假乱真的效果
- 每次训练判别器会得到加强，而生成器也会得到加强，相互拮抗，都能够得到进步

## cifar10_cnn.py
**利用卷积神经网络训练cifar10数据集**

layer | output_shape 
:----:|:-----: 
input | (None, 32, 32, 3) 
conv2d | (None, 32, 32, 32) 
activation | -
conv2d | (None, 30, 30, 32) 
activation | (None, 30, 30, 32)
maxpooling | (None, 15, 15, 32)
dropout |  -
conv2d | (None, 15, 15, 64)
activation | -
conv2d | (None, 13, 13, 64) 
activation | -
maxpooling | (None, 6, 6, 64)
dropout | -
flatten | (None, 6 x 6 x 64 = 2304)
dense | (None, 512)
activation | -
dropout | -
dense | (None, 10)
activation | -

优化：RMSprop
loss函数：categorical_crossentropy

## minst_transfer_cnn.py
**迁移卷积神经网络：一篇有关迁移学习的论文提到，只要改变最后的全连接层（之前的都为瓶颈层）即可实现迁移学习，可以快速收敛。本例子是利用手写数字前5位数训练模型，之后冻结全连接层前的所有层，将模型应用于后5位数字的训练**
1. 准备手写数据
2. 将模型、训练集的输入及其标签作为形参，构建损失函数和优化器，并输出accuracy
3. 将手写数据分成两部分，第一部分所有图片及标签是0-4，第二部分为5-9
4. 构建瓶颈层网络，构建最后的全连接层网络
5. 用model = Sequential(*layer1* + *layer2*)连接网络，之后用0~4的手写数字集进行训练
6. (l.trainable = False for l in *layers*)冻结瓶颈层
7. 用5中连接后的模型再次训练5~9的手写数字，由于只有全连接层可以训练，所以参数大大减少，很快就可以收敛

**迁移学习保留瓶颈层，是因为瓶颈层是在进行特征提取，而全连接层进行特征的组合。对于某些监督学习，特征差异较大，可以之后再连接可训练的卷积层（而不是像本例中只连接可训练的全连接层）进行**
- **模型**

layer | output_shape 
:----:|:-----: 
input | (None, 28, 28, 1)
conv2d | (None, 26, 26, 32)
activation | -
conv2d | (None, 24, 24, 32)
activation|-
maxpooling | (None, 12, 12, 32)
dropout| -
flatten | (None, 12 x 12 x 32 = 4608)
以上为瓶颈层|以下为迁移学习后可训练层
dense | (None, 128)
activation | -
dropout | -
dense | (None, 5)
activation | -
 
## deep_dream.py
**用keras实现deep dream的效果，一开始需要下载模型的权重数据**
1. 设置命令行参数，分别为图片地址和转换结果的文件名
2. 用路径名载入图片数据，转为numpy array的格式，并进行预处理转换(像素除以255，减去0.5，乘以2)
3. 用model = inception_v3.InceptionV3(...)载入模型
4. 用dream = model.input设置数据的输入点
5. 用from keras import backend as K， loss = K.variable(0.)创建一个浮点数占位变量，之后在每一层的输出中，计算均方差，其和记为loss
6. 用K.gradients(loss, dream)计算根据loss得到的梯度，并正则化，将这些计算封装为函数，之后使用。设置一个resize_img函数，对图片进行缩放，成为指定大小的图片
7. **设置梯度上升函数，这是deepdream的关键**，当loss大于某个值之前，将梯度增加在图片上（正好是图像识别训练的逆过程，梯度下降，并在loss足够小后结束）
8. loss足够大时结束，输出图片

**这样的一个过程相当于把模型的权重（即分析图片得到的特征）返回到图片上，是训练的逆过程，所以会产生非常奇异的结果**
