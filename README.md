<!-- vscode-markdown-toc -->
* 1. [通用](#)
	* 1.1. [迁移学习](#-1)
	* 1.2. [工程上的技巧](#-1)
		* 1.2.1. [验证集的使用](#-1)
		* 1.2.2. [参数搜索](#-1)
		* 1.2.3. [交叉验证](#-1)
* 2. [ 2. CV相关](#2.CV)
	* 2.1. [理解风格化 neural style transfer 1](#neuralstyletransfer1)
		* 2.1.1. [模型前向传播流程例子](#-1)
		* 2.1.2. [其他](#-1)
* 3. [NLP工程相关](#NLP)
	* 3.1. [NLP 任务](#NLP-1)
		* 3.1.1. [基于词的embedding的任务](#embedding)
		* 3.1.2. [基于特征提取的任务](#-1)
	* 3.2. [embedding的ngram2vec预训练](#embeddingngram2vec)
		* 3.2.1. [requirements](#requirements)
		* 3.2.2. [流程](#-1)
	* 3.3. [使用预训练好的词的embedding进行CNN分类](#embeddingCNN)
		* 3.3.1. [使用ngram2vec中的embedding作为模型预训练权重进行文本分类1](#ngram2vecembedding1)
* 4. [NLP网络框架相关](#NLP-1)
	* 4.1. [理解attention的image to caption（图片的文字描述）](#attentionimagetocaption)
		* 4.1.1. [一、一个简单模型](#-1)
		* 4.1.2. [二、增加Attention](#Attention)
		* 4.1.3. [详细过程：](#-1)
	* 4.2. [4.2理解bert](#bert)
		* 4.2.1. [简要笔记（等待更新）](#-1)
* 5. [GAN](#GAN)
	* 5.1. [GAN的简单实现方式dcgan-mnist](#GANdcgan-mnist)
	* 5.2. [进阶-acgan](#-acgan)
* 6. [encoder-decoder和auto encoder相关](#encoder-decoderautoencoder)
	* 6.1. [变分自编码器（VAE）初步](#VAE)
* 7. [强化学习相关](#-1)
	* 7.1. [理解DQN玩flappy bird](#DQNflappybird)
* 8. [其他](#-1)
	* 8.1. [deep_dream](#deep_dream)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->



# AI_Storage

<!--如何生成目录：在vscode中安装Markdown TOC 1.4.0，然后command/ctrl + shift + P，输入generate TOC，点击即可 -->



<!-- 功能
- 记录我在AI算法工程方面的知识点、经验和学习笔记
- ASAP(as short as possible)：尽可能用简短的几句话完成描述，突出重点
来源
1. 书籍心得笔记的浓缩
2. 源码阅读的部分注释
3. 自己其他项目的记录
4. 自己写的教程 -->

<!-- # temp
TODO:先把这个项目中的文档精简更新到这里，然后从其他项目中merge，merge后最好删除它们
TODO：从nlp private中更新笔记
TODO：AI.Learning.Notes.III.Keras.II中很多RNN lstm内容没有更新到这里，需要重新理解
TODO：AI.Learning.Notes.RNN.I也是
TODO：浏览自己的所有其他private public项目，移动相应的文档到这里，然后删除他们，比如tf lite， tfjs等工程化相关内容
TODO：增加bert的相关理解，从transformer等开始
-->


##  1. <a name=''></a>通用
- 使用神经网络进行线性拟合：使用Dense或nn.Linear从n个feature连接到1个输出，线性激活函数，RMSE作为loss
###  1.1. <a name='-1'></a>迁移学习
- 例子：keras拿取一个CNN模型，使用model = Model(inputs=[XXX],outputs=[XXX])抽取局部网络计算图做特征提取，然后使用其他softmax或回归层连接到这层特征，使用l.trainable=False for l in layers来固定特征提取层的权重，只训练后边的分类/回归层。
- 作用：分步优化以减少计算量（比如NLP中先embedding再下游任务，而不是边embedding边下游任务）；作为已经训练好的的特征提取器（如CNN中底层的直线检测的Filter很通用，可砍掉最后几层加分类回归任务，或者作为encoder）
- 需要注意抽取的层的位置，如果高阶特征相似可以抽较多的层，高阶特征不相似就抽取更少的层，然后后面加层构建高阶特征。
###  1.2. <a name='-1'></a>工程上的技巧
####  1.2.1. <a name='-1'></a>验证集的使用
- 在keras中训练时划分一定比例为验证集，同时增加EarlyStop的Callback，这样在训练时，如果在验证集上loss连续增加某几个步数之后，会EarlyStop停掉 
####  1.2.2. <a name='-1'></a>参数搜索 
- 使用f1 score搜索2分类的softmax阈值
####  1.2.3. <a name='-1'></a>交叉验证
- 多使用K5，K10交叉验证代替通常的训练/验证/测试集划分




##  2. <a name='2.CV'></a> 2. CV相关
###  2.1. <a name='neuralstyletransfer1'></a>理解风格化 neural style transfer 1
- 简述：使用风格图片和被风格化图片的中间特征的范数作为loss优化被风格化图片实现风格化
- 原文档@./AI.Application/XXXStylizePicturesXXX
####  2.1.1. <a name='-1'></a>模型前向传播流程例子
1. 图片预处理
2. 加载VGG模型以及权重，拿出relu1_1,relu2_2,relu2_1等层备用
3. 构造一个将content图像输入，relu1_1层输出的计算子图
4. 构造一个将被风格化图片作为输入，relu2_2层输出的计算子图
5. 将上两个计算图的输出特征用相减的2范数的计算图节点连接，得到范数
6. 将范数作为loss，所有层的权重都固定，只留下被风格化图片是Trainable的
7. 优化以minimize loss，此时只有被风格化的图片被优化，逐渐使得其提取的特征接近风格化图片的特征
####  2.1.2. <a name='-1'></a>其他
- 这种风格化方法每次风格化都是完全的训练过程，耗时很大，不如类似自编码器结构的网络
- 抽取多深的特征层来求loss比较重要，抽取风格化图片接近底层的特征来求loss时，会使得风格化比较局部，可以看成是每个很小的点和线的风格的替换






##  3. <a name='NLP'></a>NLP工程相关
###  3.1. <a name='NLP-1'></a>NLP 任务
####  3.1.1. <a name='embedding'></a>基于词的embedding的任务
- 词的embedding预训练：使用w2v，ELMo等进行监督或无监督的学习，对词的embedding进行预训练（类似CNN的完整的训练，或者训练自编码器提取初级特征） 
- 下游任务和fine tuning：在得到了一个好的预训练的词的embedding，将embedding载入到模型中，Frozen掉embedding的权重，进行下游任务，然后fine tuning非embedding层的权重（类似CNN使用已经训练好的模型迁移学习优化最后几层，或者拿出自编码器的编码层接上下游任务层）
####  3.1.2. <a name='-1'></a>基于特征提取的任务
- 对于bert等模型，它不仅仅包含词的embedding，也包含句子的embedding和position embedding，因此训练时会采用额外的任务，比如训练bert中词的embedding使用mask，训练句子是采用两个句子是否是上下文的分类任务进行预训练
- 预训练过后，并非是直接拿出词、句子和position的embedding使用（当然也可拿出词的embedding作简单的下游任务）。而是拿出Transformer encoder的最后一层，作为句子的特征（一个由每个词一个bert dim向量构成的矩阵）
- 然后用上面提取的特征作下游任务，相当于bert是一个encoder，从句子到矩阵
###  3.2. <a name='embeddingngram2vec'></a>embedding的ngram2vec预训练
####  3.2.1. <a name='requirements'></a>requirements
- ngram2vec，包含ngram以及w2v和glove的c代码(运行前编译): https://github.com/zhezhaoa/ngram2vec
- 预训练w2v：https://github.com/Embedding/Chinese-Word-Vectors
####  3.2.2. <a name='-1'></a>流程
1. 可选：使用ngram分词获得词的pair，用于扩充词库，得到vocab文件
2-1. 如果使用gensim的w2v，glove等进行训练，可以从ngram的vocab文件中提取出所有的词作为jieba分词的userdict，加载自定义词典分词，然后训练，也可不加载自定义词典
2-2. 也可使用ngram2vec中编译好的二进制文件直接训练，需要根据ngram2vec文档准备相应的文件，参考ngram2vec的流程图和运行的.sh文件中记录的流程
4. 导出w2v字典，使用相似度或类比等方法评估

###  3.3. <a name='embeddingCNN'></a>使用预训练好的词的embedding进行CNN分类
####  3.3.1. <a name='ngram2vecembedding1'></a>使用ngram2vec中的embedding作为模型预训练权重进行文本分类1
1. 使用ngram2vec提供的方法(例如load_dense)载入https://github.com/Embedding/Chinese-Word-Vectors中的预训练模型，拿到词汇表vocab\["w2i"\].keys()，写入文件
2. 1.中的文件作为jieba分词的userdict，使用jieba进行分词
3. 使用1.中的vocab\["w2i"\]把文章分词过后的全部词变成index的list
4. 输入大小需要一致，作出文章中词数目的分布直方图，选取一个值作为目标dim
5. 通过0补和截取的方法将index的list变成同样大小的，最终变成一个矩阵，这个矩阵是NN的输入，shape为(n_sample,dim)，可使用keras.preprocessing中sequence的pad_sequence进行
6. CNN之前添加一个Embedding层，这里我们不优化embedding，而是用现有的embedding直接作为Frozen的权重，以keras的embedding层为例，input dim是词汇表大小，output dim是embedding过后的向量大小，input length是句子最大词的数目（也就是上面的dim）
6.5 CNN会采用3个不同大小的卷积核，以及较多的filter进行卷积，最终在每个filter上得到的feature map中取最大值，将feature map变成一个值，从而得到特征向量，取最大值也可以实现对可变输入的扩展，这个固定的特征向量通过softmax完成分类任务
7. 这个embedding层使用weights=\[embedding_matrix\]来添加预训练好的embedding结果
8. 训练，注意使用keras的EarlyStop功能，训练集，验证集和测试集比例7:1:2，验证集用于验证什么时候停止训练
9. 完成之后封装模型，封装预处理pipeline，构造函数输入string，输出分类的softmax概率

  
##  4. <a name='NLP-1'></a>NLP网络框架相关
###  4.1. <a name='attentionimagetocaption'></a>理解attention的image to caption（图片的文字描述）
- 代码+注释 位置:@./AI.Learning/ImageCaptionWithAttentionModels.py
####  4.1.1. <a name='-1'></a>一、一个简单模型
1. Encoder:使用预训练的CNN进行fine tuning，结束后截取出输入Image到一个
feature map flatten成的向量或者直接得到的特征向量的输出，
例如Height*Width*3的图片到2048*14*14的向量
2. Decoder:decoder在第一次时会输入Encoder给出的图片特征以及
和\<start\>词向量一起concat过后的向量，输入到LSTM预测下一个词，
第一次过后每次LSTM就输入上一次得到的词的embedding，
输出下一个预测的词的向量，通过LSTM的输出然后softmax，
得到输出词的概率，转变成词语，直到到达end标记。
在此过程中，LSTM自己的隐藏层每次输出下一个h向量，然后下一步将上一步输出的h向量作为输入（RNN的特性）
####  4.1.2. <a name='Attention'></a>二、增加Attention
Decoder发生变化：    
1. 原来LSTM每次输入上一个词的embedding，变为输入上一个词的embedding拼接上encoder给出的图片向量  
2. 但是每次拼接的encoder向量都是一样的，没有意义，于是使用Attention来修改这个向量，使其在一部分中有重点  
于是Attention出现  
Attention是给encoder结果加权重的，输入encoder结果以及LSTM的decoder输出的上一个结果，输出加了权重的encoder结果  
Encoder的输入（图片），以及Decoder的输出（词的onehot）都是明确的，而Attention如何优化，如何给出Image decode过后的权重，是需要关注的
####  4.1.3. <a name='-1'></a>详细过程：
1. 输入Image，经过预训练的CNN得到feature map，作为encoder out，这个过程可能需要先通过迁移分类任务fine tuning后面几层
2. encoder out将作为LSTM的内部权重h的初始（由于feature map是2维的，通过mean转成向量传入LSTM）
3. LSTM的输出的decoder向量，将会经过softmax到vocab的大的维度，给出每一个词的概率（dim就是词库中词的数目）
4. LSTM的输入是词向量（来自于上一个LSTM预测的词的embedding，或者初始\<start\>的embedding）再拼接经过attention的feature map
5. LSTM输出的decoder除了用于3中预测词，还用于给feature map加上attention，（用于LSTM的下一次显式输入）
6. 给feature map加上的attention是和feature map同样长宽的，但是只有1个通道，里面的每个值都是softmax的结果
7. 为了得到这个softmax，首先feature map的n通道通过Dense变为attention dim通道的特征，然后将这个特征与decoder向量经过Dense得到的attention dim长度向量的特征相加，最后Dense到1，然后softmax输出
8. 最终输入Image，每次输出词的softmax，经过argmax得到词，直到得到\<end\>

###  4.2. <a name='bert'></a>4.2理解bert
####  4.2.1. <a name='-1'></a>简要笔记（等待更新）
- 需要先了解Transformer encoder，以及self-attention，
- 包含3个embedding，词的embedding，句子的embedding和position embedding，为了训练这些embedding需要特定的非监督任务
- 拼接上述embedding，然后经过较多个Transformer encoder，最终针对一个句子输出sequence\_length x bert\_feature_dim 大小的矩阵，sequence\_length是句子中词的数目，每个词有一个bert feature dim大小的向量
- 使用bert输出的向量的第一个词的向量，去掉其他剩下所有词的向量，可以用于分类（作为一个句子的特征向量）
- 如果使用bert全部的向量用于下游任务，可以采用CNN处理（将输出当作feature map），可采用LSTM等RNN框架按照顺序每次输入一个feature vector处理


##  5. <a name='GAN'></a>GAN
###  5.1. <a name='GANdcgan-mnist'></a>GAN的简单实现方式dcgan-mnist
- 原文档@./AI.Learning.Notes.I
- 鉴别器:图片作为输入，softmax输出label为True的概率。
- 生成器：100长度噪声向量作为输入，输出和鉴别器输入相同shape的图片
- 鉴别器优化：从噪声中生成的图片需要被鉴别器判定为False，而真实图片判定为True（loss为二分类loss）
- 生成器优化：固定鉴别器权重，优化生成器使得生成的图片被鉴别器判定为True（loss为被鉴定器判定为False的概率）
- 鉴别器和生成器交替优化，形成对抗

###  5.2. <a name='-acgan'></a>进阶-acgan
- dcgan无法生成指定label的图片，现在改进希望能够指定生成图片的label
- 改进：鉴别器不仅仅鉴别图片的真假，还鉴别图片的label
- 改进：生成器添加一个label的特征，loss拆成两部分，一部分是使得鉴别器鉴别为True的loss，另一个部分是使得鉴别器鉴别为预设的输入label的loss
- 例如：生成器输入随机数，concat一个label 0010（第3类），此时loss除了鉴别器真假的loss，还有一个是被鉴别器识别为第3类的loss，有两部分loss



##  6. <a name='encoder-decoderautoencoder'></a>encoder-decoder和auto encoder相关
###  6.1. <a name='VAE'></a>变分自编码器（VAE）初步
- encoder：可以使用CNN结构，输入图片拿到特征
- encoder给出的特征将通过全连接同时连接到两个值中
- 这两个值分别作为高斯分布的均值和方差（也就是encoder最终从一个图片得到高斯分布）
- 按照这个均值和方法采样生成满足分布的随机数
- 将这个随机数作为decoder的输入，decoder将其复原成encoder的原始输入
- 使用时，可以砍掉encoder层，直接输入高斯的均值和方差，然后生成原始图片输入，连续地改变均值和方差，能够发现生成的图片有连续的过渡

##  7. <a name='-1'></a>强化学习相关
- 较多的工程化经验（自动驾驶领域强化学习），文档和代码可以参考我的开源项目：<https://github.com/B-C-WANG/ReinforcementLearningInAutoPilot>
###  7.1. <a name='DQNflappybird'></a>理解DQN玩flappy bird
- 模型的输入state是None,80,80,4的4帧黑白图像，输出是跳/不跳的回归值，这个值是预测的Q值，也就是NN在这里是构建一个Q表
- 跳和不跳的Q越大，表明该state下采用此action的期望收益越大，一般在使用阶段采用贪心算法选Q最大的，而在训练阶段使用e-greedy一定概率选择Q最大或随机的（其他算法，比如DDPG，A3C等有些是输出概率，就能够直接概率决定action，有些是输出浮点数，就是加噪声）
- Q值的获取根据人为设置的reward来，比如穿过管子reward是10，死亡是-10，其他为1等等
- 通过每次的game over时的reward序列求出目标Q值，作为模型的标签对其进行优化




##  8. <a name='-1'></a>其他
###  8.1. <a name='deep_dream'></a>deep_dream
- 找一个CNN网络，砍掉最后分类的几层，输入图片，输出特征层，将特征的范数作为loss，然后对输入图片求梯度，将梯度变为负（梯度上升），这样的话图片的特征层的范数反而不断上升，上升到一定程度输出图片