## babi_memnn.py
**用记忆神经网络训练bAbI数据集**
- bAbI数据集下载到一半时如果取消，会得到不完整的文件而报错，此时应当在代码中打印出下载地址，如`path = get_file(...) print(path)`，然后删除这个文件，不过一般在home目录下，windows在用户目录下
- 在分析代码时，先直接看模型summary以及喂入的数据集，之后返回看数据集是怎样得到的。
1. 函数`tokenize(sent)`：输入一个句子，将句子的每一个词和符号作为元素存储在list中
2. `parse_stories(lines, only_supporting)`：only_supporting为True时，只有支持答案的句子才会出现。这个函数基于`tokenize`，针对bAbI的数据集，输入行，得到单词的list
3. `get_stories()`：基于`parse_stories`，输入文件名，解析出句子中词语的list，比如某个输出：`[(['Mary', 'moved', 'to', 'the', 'bathroom', '.', 'John', 'went', 'to', 'the', 'hallway', '.'], ['Where', 'is', 'Mary', '?'], 'bathroom'), 每个set是一个完整的问题，含有两个list和一个string，前一个list是描述，后一个是问题，最后一个string是答案
4. `vectorize_stories()`直接转换为词向量，一个单词具有独一无二的序号
5. 描述的最大长度为68个单词，问题的最大长度为4个单词，用来训练的数据为10000组，用来测试的数据为1000组
6. 所采用的问题集都是关于空间位置逻辑的，比如“a在b处，b在c处，问d在哪？”因而，描述的shape为(None, 68)，多为[0,0,0,...,18,n,1]，因为问题结尾多为"the ... ."。问题的shape为(None, 4)，几乎都为[7,13,n,2]，对应于"where is ... ?"，而答案的shape为(None, 22)，是22维的独热编码（整个数据集的词汇量为22个，多为“bedroom, hallway, bathroom”等空间位置词汇，以及人名）

**模型结构**
模型结构包含并联层，在layer前用n-m-l注明，n为层的深度，m，l等等为同一个串联层的标志，在这里m=1代表描述的输入，m=2代表问题的输入。layer后面的括号中代表了对哪个层进行了运算(省略了dropout和activation层)

layer | shape
:-- | :--
1-1, story_input | (None, 68)
1-2, question_input | (None, 4)
2-1-1, story_encoder_m, embedding(1-1) | (None, 68, 64)
2-1-2, story_encoder_c, embedding(1-1) | (None, 68, 4)
2-2, question_encoder, embedding(1-2) | (None, 4, 64)
3, dot(2-1-1, 2-2)| (None, 68, 4)
4, add(3, 2-1-2)| (None, 68, 4)
5, permute(4) | (None, 4, 68)
6, concatenate(5, 2-2) | (None, 4, 132)
7, LSTM(6) | (None, 32)
8, dense(7) | (None, 22)

之后输出的(None, 22)和结果和真实答案(None, 22)进行损失计算。

![](http://i.imgur.com/Ur5rPDe.png)


## babi_rnn.py
**数据处理与上一例类似，只不过改变了数据集，描述的最大长度为552，问题最大长度为5，答案的维度（总也就是词汇量）为36**
**模型结构**

layer | shape
:-- | :--
1-1, story_input | (None, 552)
1-2, question_input | (None, 5)
2-1, story_encoded, embedding(1-1)| (None, 552, 50)
2-2, question_encoded, embedding(1-2) | (None, 5, 50)
3-2, recurrent.LSTM(2-2) | (None, 50)
4-2, repeat_vector(3-2) | (None, 552, 50)
5, add(2-1, 3-2) | (None, 552, 50)
6, recurrent.LSTM(5) | (None, 50)
7, dense(6) | (None, 36)

![](http://i.imgur.com/TvunuJa.png)


## conv_lstm.py
- **这个例子首先认为创造一个移动的矩形，形成一段影片，之后根据这个影片预测出矩形之后的移动，涉及到图像处理和时间序列，因而需要ConvLSTM**
- 需要较高显存才可执行，可将n_samples调低一些，同时调整which
- **模型结构(输入为(batch_size, n_frames[帧数], width, height, channels))**

layer | shape
:--: | :--:
input | (None, None, 40, 40, 1)
conv2d_LSTM | (None, None, 40, 40, 40)
batch_normalization | -
conv2d_LSTM | (None, None, 40, 40, 40)
batch_normalization | -
conv2d_LSTM | (None, None, 40, 40, 40)
batch_normalization | -
conv2d_LSTM | (None, None, 40, 40, 40)
batch_normalization | -
**conv3d** | (None, None, 40, 40, 1)

1. 创建一个指定大小、帧率和batch_size的影片作为输入。在影片中添加3~7个正方形，设定正方形的运动起始点和运动方向
2. 创建输入数据，即创建正方形的移动影片，并在视频加上噪点，是为了防止数据为0，增强模型robust
2. 创建下一刻的数据作为y，y的影片时间为输入的时间t+1，相当于y总是快于输入一秒，这样训练能够预测，y不加入噪点
3. 用上述模型，以完整影片作为输入，影片时间+1的y作为求取loss的输出，进行训练

**预测和可视化**
1. 训练完成之后，首先用which变量选择一个sample，之后会取7帧作为输入，进行fit，得到预测之后的几帧，变量为track
2. 截取和预测片段相同的真实片段track2，将track和track2的每一帧作图在一起，输出图像进行对比

## imdb_lstm.py
利用LSTM处理IMDB数据集，IMDB是一个网络电影数据集，输入是已经转化为词向量的数据，输出为正面或负面评价，是二分类问题。
**模型**
 
layer | shape
:--:  | :--:
input | (None, 80)
embedding | (None, None, 128)
LSTM | (None, 128)
dense | (None, 1)

## imdb_cnn_lstm.py
在IMDB中使用conv1D的卷积。LSTM可以处理序列化的数据，而一维卷积则是类似于2维卷积，处理序列上相关的数据。
**模型**

layer | shape
:--: | :--:
input | (None, 100)
embedding | (None, 100, 128)
dropout | (None, 100, 128)
conv1d | (None, 96, 64)
pooling | (None, 24, 64)
LSTM | (None, 70)
dense | (None, 1)
activation | (None, 1)
## imdb_bidirecational_lstm.py
双向LSTM网络
**模型**

layer | shape 
:--: | :--:
input | (None, 100)
embedding | (None, 100, 128)
bidirectional | (None, 128)
dropout | (None, 128)
dense | (None, 1)

## mnist_hierarchical_rnn.py
多层反馈神经网络，这个例子中是采用的keras函数式建模(不同于
Model.add()，采用layer(parameters)(last_layer))。
**模型**

layer | shape 
 :--: | :--:
input-x  |  (None, 28-row, 28-col, 1)
LSTM(row) |(None, 128)
timeDirtributed(x) | (None, 28, 128)
LSTM(col) | (None, 128)
dense | (None, 10)
将(28, 1)的行向量用LSTM编码成(128)的行向量，之后将(28, 128)的图形用LSTM编码成(128)的图像向量，最后全连接获得标签。
 




## 体会
- **embedding层是专用于语言处理的层，能够降低维度，在词向量数据集IMDB中经常使用，而在图形处理采用convLSTM时，没有用到这个层。这个层的具体作用可见keras手册**
- **LSTM层在keras中的使用类似于dense层，convLSTM也可直接当做conv使用，在了解了LSTM的机理之后可以不用造轮子直接使用**

  
