
# AI_Storage

## Function
- 记录我在AI算法工程方面的知识点、经验和学习笔记
- ASAP(as short as possible)：尽可能用简短的几句话完成描述，突出重点
## Source
1. 书籍心得笔记的浓缩
2. 源码阅读的部分注释
3. 自己其他项目的记录
4. 自己写的教程

# CV相关
## 理解风格化neural style transfer


# NLP工程相关
## NLP 任务
- embedding预训练：使用w2v，ELMo，bert，GPT等进行监督或无监督的学习，对词的embedding进行预训练（类似CNN的完整的训练，或者训练自编码器提取初级特征）
- 下游任务和fine tuning：在得到了一个好的预训练的embedding过后，将embedding载入到模型中，Frozen掉embedding的权重，进行下游任务，然后fine tuning非embedding层的权重（类似CNN使用已经训练好的模型迁移学习优化最后几层，或者拿出自编码器的编码层接上下游任务层）
## embedding的ngram2vec预训练
### requirements
- ngram2vec，包含ngram以及w2v和glove的c代码(运行前编译): https://github.com/zhezhaoa/ngram2vec
- 预训练w2v：https://github.com/Embedding/Chinese-Word-Vectors
### 流程
1. 可选：使用ngram分词获得词的pair，用于扩充词库，得到vocab文件
2-1. 如果使用gensim的w2v，glove等进行训练，可以从ngram的vocab文件中提取出所有的词作为jieba分词的userdict，加载自定义词典分词，然后训练，也可不加载自定义词典
2-2. 也可使用ngram2vec中编译好的二进制文件直接训练，需要根据ngram2vec文档准备相应的文件，参考ngram2vec的流程图和运行的.sh文件中记录的流程
4. 导出w2v字典，使用相似度或类比等方法评估

## 使用预训练好的embedding进行CNN分类
### 使用ngram2vec中的embedding作为模型预训练权重进行文本分类1
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

  
# NLP网络框架相关
## 理解attention的image to caption（图片的文字描述）
- 代码+注释 位置:./AI.Learning/ImageCaptionWithAttentionModels.py
### 一、一个简单模型
1. Encoder:使用预训练的CNN进行fine tuning，结束后截取出输入Image到一个
feature map flatten成的向量或者直接得到的特征向量的输出，
例如Height*Width*3的图片到2048*14*14的向量
2. Decoder:decoder在第一次时会输入Encoder给出的图片特征以及
和<start>词向量一起concat过后的向量，输入到LSTM预测下一个词，
第一次过后每次LSTM就输入上一次得到的词的embedding，
输出下一个预测的词的向量，通过LSTM的输出然后softmax，
得到输出词的概率，转变成词语，直到到达end标记。
在此过程中，LSTM自己的隐藏层每次输出下一个h向量，然后下一步将上一步输出的h向量作为输入（RNN的特性）
### 二、增加Attention
Decoder发生变化：    
1. 原来LSTM每次输入上一个词的embedding，变为输入上一个词的embedding拼接上encoder给出的图片向量  
2. 但是每次拼接的encoder向量都是一样的，没有意义，于是使用Attention来修改这个向量，使其在一部分中有重点  
于是Attention出现  
Attention是给encoder结果加权重的，输入encoder结果以及LSTM的decoder输出的上一个结果，输出加了权重的encoder结果  
Encoder的输入（图片），以及Decoder的输出（词的onehot）都是明确的，而Attention如何优化，如何给出Image decode过后的权重，是需要关注的
### 详细过程：
1. 输入Image，经过预训练的CNN得到feature map，作为encoder out，这个过程可能需要先通过迁移分类任务fine tuning后面几层
2. encoder out将作为LSTM的内部权重h的初始（由于feature map是2维的，通过mean转成向量传入LSTM）
3. LSTM的输出的decoder向量，将会经过softmax到vocab的大的维度，给出每一个词的概率（dim就是词库中词的数目）
4. LSTM的输入是词向量（来自于上一个LSTM预测的词的embedding，或者初始<start>的embedding）再拼接经过attention的feature map
5. LSTM输出的decoder除了用于3中预测词，还用于给feature map加上attention，（用于LSTM的下一次显式输入）
6. 给feature map加上的attention是和feature map同样长宽的，但是只有1个通道，里面的每个值都是softmax的结果
7. 为了得到这个softmax，首先feature map的n通道通过Dense变为attention dim通道的特征，然后将这个特征与decoder向量经过Dense得到的attention dim长度向量的特征相加，最后Dense到1，然后softmax输出
8. 最终输入Image，每次输出词的softmax，经过argmax得到词，直到得到<end>
