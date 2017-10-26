## lstm_text_generation.py
基于LSTM的文本生成。采用的数据集是尼采的著作，可以采用其他语料库进行替换生成（如中文、学科论文、笑话语料库等）
- 数据库载入和预处理
 - 读取数据
 - 将数据中的字符(chars)提取，如"\n", "a", ")"和空格等
 - 构造字符和向量互换的两个词典，如{1: "a", 2: "b"}和{"a": 1, "b": 2}
 - 建立x和y数据集，x是一个由maxlen个**字符**组成的句子，而y为该句子紧接着的下一个字符，其中每次取了40个字符之后就跳过3个字符，如x1=(0,40),y1=(41)和x2=(3,43),y2=(44)这样类似于conv中的步长，由于例子中步长为3，所以最终所得样本数量为原来字符数量的1/3
 - 得到了x,y之后，对于每一个字符组成的句子，建立一个长度为`len(chars)`的数组，采用独热编码表示单词，如abc就是[1,1,1,0,0,0,0...0]
 - 最终得到的shape为x: (200285, 40, 58), y: (200285, 58)，200285是上个步骤数据集生成的句子，在一次训练中只使用batch_size训练其中一小部分(本例子中batch_size为128)，**所以上面的shape通常也记为(None, 40, 50)，None是batch_size**
-  模型

layer | shape 
:--: | :--:
input | (None, 40, 58)
LSTM | (None, 128)
dense | (None, 58)
activation | -


- 训练和生成

 1. 例子中训练和生成时同时进行，总共有60次训练，每次喂入128个数据进行
 2. 每次训练结束后就测试，先选一个随机起点，得到40长度字符的句子，之后将字符转化为独热编码向量,shape为(40, 58)
 3. 用model.predict进行预测，输出为(58)的矩阵，然后用激活函数以及argmax求得概率最大的输出，用字符词典转换成相应的字符，就是最后预测的结果
 4. 每次predict得到一个字符，把它加进原来的句子中，然后重新取后面40个，相当于滑动窗口，这样能够基于生成的句子来预测（前面由于是原句，所以生成会较为合理，后面很多是生成的句子，不合理性较高）
 5. 用了一个循环语句，重复3~4过程，生成400个字符，组成句子 
 6. 用sys.stdout.write进行输出


**心得和展望**
- 以字符为单位进行预测，而不是以词语为单位，这样减少了很多处理量，而且还能够完整地生成出单词，也许这是LSTM的优势所在
- 用学术论文作为数据集进行训练，之后根据自己写的内容进行预测，以此达到论文润色的目的，需要关注低频词汇的影响。

## lstm_fasttext.py
**利用fast text进行text的分类**
1. create_ngram_set函数：输入一个list和int，将list转化为窗口为int，步长为1的元组列。create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)——[(4, 9), (4, 1), (1, 4), (9, 4)]。create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)——[(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
2. add_ngram函数：将输入的sequences按照create_mgram_set的方式进行处理，处理得到的输出用token_indice进行映射，转换成另一个int，**需要注意这里采用的是ngram_range而不是ngram_value，如果ngram_range=4，则相当于ngram_value=1,2,3,4的情况都要进行转换**
3. 载入IMDB数据，长度为20000，得到的数据是二维嵌套数组，每一个数组中是一个句子的词向量构成的数组
4. **如果ngram_range大于1**，对于从2到ngram_range的int，都采用1. 中的函数进行转化，然后将所得得到set更新至ngram_set中(直接用`<set>.update(<set>)`)，完成过后，得到了一些含多个词的词库，对于固定搭配很有用。比如，ngram_range=1，则ngram_set为((1),(2)...)，如果ngram_range=3，则ngram_set为((1),(2),(1,2),(1,2,3))，之后建立字典，在已有词的情况下（载入的数据在ngram_value=1的情况下已经实现了转换为词向量），在后面追加ngram_value=2,3...直到ngram_range词语数量对应的映射。
5. 建立的词库即为token_indice，用add_ngram借助token_indice进行转化
6. 之后进行训练，模型：

layer | shape
:--: | :--:
input | (None, 400)
embedding | (None, 400, 50)
global_average_pooling1D | (None, 50)
dense | (None, 1)



**笔记**
- **这个例子的特殊之处在于，以往的处理都是一个word一个向量，这里添加了ngram_value，实现了多个word一个向量，尤其在中文处理中非常有用。现将x_train中的原有词向量保留(如(1)->(1),(2)->(2))，之后添加2个word形成的向量，如((1,2)->(101))，再添加3个word((1,2,3)->(1001))...x_test也如此转化**

## stateful_lstm.py
对于keras，在LSTM将stateful设为True，可以记住较长序列。
1. 生成初始数据：输入：一个振幅衰减的cos函数，shape为(50000, 1, 1)。输出： 一个先于输入几个单位的输出，相当于时间上是输入的未来，用于预测，用lahead控制先几个单位
2. 训练，模型：

layer | shape
 :--: | :--:
total_input | (50000, 1, 1)
batch_input | (25, 1, 1)
LSTM | (25, 1, 50)* (25 is batch_size)*
LSTM | (25, 50)
dense | (25, 1)
batch_output | (25, 1)
total_output | (50000, 1)
trainable params: | 30651
**训练完成之后，对输入进行predict，得到输出和经过lahead的输出放在一起作图对比，下面分别是lahead为1,20和5000**
![](http://i.imgur.com/raCciYR.png)
![](http://i.imgur.com/cHaGURR.png)
![](http://i.imgur.com/iBzntPb.png)

## mnist_irnn.py
IRNN："A Simple Way to Initialize Recurrent Networks of Rectified Linear Units"
模型：SimpleRNN中设置了较多的参数

layer | shape 
:--: |:--:
total_input | (60000, 784, 1)
input | (None, 784, 1),batch_size=32
SimpleRNN | (None, 100)
dense | (None, 10)
activation | -
output | (None, 10)

## lstm_benchmark.py
keras的LSTM设置了三种implement:
- implement=0（默认设置），对输入进行预处理，计算更快，峰值内存更大
- implement=1，不对输入进行预处理，计算慢，内存消耗更小
- implement=2，将输入、输出和遗忘门的权重连接在一起，使GPU计算更快，以减少正则化为代价。

这三种方式的使用需要根据机器和数据集决定。
例子中对于作图数据的获取，是采用result[0].epoch得到每一个epoch的accuracy，或采用result[0].history
![](http://i.imgur.com/ngyqiW4.png)


## reuters_mlp_relu_vs_selu.py
激活函数ReLU和SeLU的对决，由于缺少模块，没能成功运行。两个模型是在dense层之后添加了ReLU和SeLU激活函数进行对比。
在建立了模型函数之后，给函数添加实参的方式，这里采用了function(**<dict>)的方式，用一个字典喂入实参，字典的key是实参名称，value是对应的值

## mnist_sklearn_wrapper.py
**建立CNN网络模型并利用sklearn的GridSearchCV寻找最佳模型**
-  导入手写数字数据
-  建立一个模型，其中dense_layer_size，卷积的filters和kernel_size以及pool_size作为形参
-  dense_size用一个list来存储，例子中选择了多个dense_size，如32, 64, [32, 32], [64, 64]
-  用`my_classifier = KerasClassifier(make_model, batch_size=32)`建立分类器
-  `validator = GridSearchCV(my_classifier, param_grid={...})`设置参数，其中参数可以是一个list，**如果为list，就会在之后选择出最佳参数**
-  `validator.best_params_`可以判断出最佳参数
-  `best_model = validator.best_estimator_.model`得到模型

**在这次训练中，epochs为[3]和[6]，两个值，而dense层有4个值，将根据可选的组合进行穷举搜索，得到最佳的模型。得到的最佳结果为，epoch=[6]，dense_layer_size=[64, 64]**

## mnist_siamese_graph.py
**准备**
1. euclidean distance函数，计算欧式距离，输入为两个多维向量
2. contrastive_loss，计算对比损失，输入为真实标签和predict的标签
3. create_paris函数，
4. 建立模型和accuracy计算函数

**执行**
1. 首先对于y_train和y_test，分别得到0,1,2,3...,9这10个数在原输入中的位置的矩阵，如0: (1, 9, 16...); 1:(0, 10, 13...)，结果为*digit_indices*
2. 利用create_pairs进行处理：
 - 首先求得1. 中得到的10个数字的位置矩阵的长度的最小值*n*（每个数字的位置矩阵长度是不同的，因为出现频率不同，这里求得最小值）
 - 对于每个数字的位置矩阵从0到10，对于矩阵中的index从0到*n*，得到相邻前后两个位置，然后在x_train或x_test中寻找这些位置对应的784长度的图片数据，**最终得到的就是代表同一个数字的位置上相邻的两张图片的信息**
 - 之后从1-9随机选择一个数，加上原来的d，对10取余，**这样得到的就是不同于所选数字的其他数字**，比如当前是0的话，这样得到的就是1-9，如果当前是1，得到的就是0或2-9。然后把对应的图片数据加上去
 - 对于label标签，加上[1, 0]
 - **也就是说，输入一开始是连续两张代表0的图片，接着连续两张随机代表1-9的图片（非0）；之后连续0的图片数量到达*n/2*过后，是连续两张代表1的图片，接着连续两张代表0或2-9的图片（非1）...如此循环直至每个手写数字的图片都加上。而标签label，就是[1,0,1,0,1,0...]循环**
3. 对模型喂入数据集，首先看model.fit，可以看出上面2. 创建的shape为(None, 2, 784)的数据集将分为两个(None, 1, 784)的数据集进行喂入，**第2n次喂入的两个，是手写数字确定的两个pair，第2n+1次喂入的，是手写数字随机的两个pair。比如，第一次是两个0的(None, 784)x2，第二次是两个1-9数字的(None, 784)x2，第三次又是两个0的(None, 784)x2，对应的标签：第2n个为1，第2n+1个为0**
4. **两个成对的0或者两个成对的随机输出拆分开来，**喂入同一个dense模型当中，分别得到(None, 128)的输出，输出随后经过用欧式距离函数封装的distance这个层进行运算，属于自定义的层，用Lambda定义计算方式，output_shape得出输出的shape。欧式距离的计算使用K.进行，对应于theano或tensorflow的运算。最终得出的结果的shape是(None, 1)，**也就是说一个pair得到一个值**
5. **最终得到完整的模型：输入shape为(None, 784)x2的图片pair，输出欧式距离(None, 1)，其中这None个输入，为2n和为2n+1属于不同pair**

**同种手写数字，其欧式距离是一定的，这个例子就用了这样的技巧：将某种手写数字连续排列，其标签为1，将不同于这种手写数字的其他数字随机排列，标签为0，这样进行了区别**，这种方式没有使用卷积，但是求取欧式距离的时候有与二维空间相关的计算
**流程简图：**
![](http://i.imgur.com/bGlsOmi.png)


## 关于keras模型建立和训练的部分总结
- 模型建立可以采用model=sequence()，之后使用model.add，或者直接使用函数式连接:<layer>(parameter)<last_layer>
- 关于输入节点的建立，可以直接在第一层中写入input_shape=...此时shape是不包含batch_size的，比如shape=(a, b)实际上建立的是(None, a, b)。当然也可以使用model.input()
- 模型建立完成之后，需要用model.compile()设置损失函数和优化器，也可以使用自定的loss节点作为loss函数
- 进行训练采用model.fit()，此时往里面加入参数，可以手动加入每个batch_size的参数，也可以将数据集喂入，然后设置batch_size。关于epoch，可以手动用for循环实现，也可直接在参数中设置epoch
- 用model.predict进行预测