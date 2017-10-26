## Al.Application.StylizePictures.I

- **利用神经网络进行风格化图片**
- 所用代码：https://github.com/anishathalye/neural-style

### 使用
- 参考cmd命令：
`python neural_style.py`（运行脚本） 
`--iterations 5000 `（迭代次数）
`--content example.jpg` （被风格化图片）
`--styles style.jpg` （提供风格图片）
`--output result.jpg`  （输出图片名称）
`--checkpoint-iterations 50` （每多少次迭代输出一个check图片）
`--checkpoint-output temp%s.jpg`（check图片的名称，需要带有%s）
- 效果
![](http://i.imgur.com/wRz7oX2.jpg)
![](http://i.imgur.com/4XVIZdo.jpg)

### 代码理解
#### neural_style.py
- 设置基本参数，以及VGG的路径（需要下载imagenet-vgg-verydeep-19.mat）
- 在build_parser中设置命令行参数
- 主函数中，首先获取命令行参数：
  - 读取VGG路径
  - 读取content图片
  - 读取多个style图片
  - 根据width，resize content image
  - 根据style_scale， resize style image
  - 根据style_blend_weights设置不同style图片所占的权重
  - 根据initial设置初始化图片
  - 检测ckpt的文件名设置
- 最后用一个for循环，**传递所有参数到stylize.py中的stylize函数中**，返回iteration和image，如果有ckpt设置，就会周期性地输出image，并重命名，存储图片

#### vgg.py
vgg.py是stylize的基础，它有以下5个函数：
- net(data_path, input_image)：输入VGG模型权重所在路径，以及图片，输出网络节点以及正则化均值
  - 建立一个layers的set用于建模
  - 载入VGG模型数据，得到其中的正则化以及layers的权重数据
  - 对于建立的set中的每一个元素，检测其名称，如果是conv卷积层，就载入相应的kernels和bias数据，并将kernels和bias的维度转化为符合tensorflow，如果是relu或pool层，不载入数据
  - 递归建立节点，并将节点用字典存储（**对于每一个节点，这里存储的都是从一开始到这个节点的运算，而不是接着上一个节点的运算**）
  - 返回节点和正则化均值
- _conv_layer：建立卷积层，其中weight和bias的参数都是从VGG中来，步长为(1,1,1,1),padding="SAME"
- _pool_layer：建立池化层，k_size=(1,2,2,1)，步长(1,2,2,1)，padding="SAME"
- preprocess:image数据减去均值，用于预处理，去中心化
- inprocess：image数据加上均值，还原


#### stylize.py
- 设置所涉及的layers，接着是stylize函数中的过程：
- **前向传播计算content特征**
 - 图片reshape为(1, image.shape)
 - 将VGG和图片数据传入上面的VGG.net，得到网络节点以及正则化均值数据
 - 对content图片进行正则化，得到content_pre(这里的处理都是采用placeholder，但是由于之后只会feed content_image的数据，所以这里直接描述content图片)
 - net[CONTENT_LAYER]截取VGG中从一开始到relu4_2这一层的节点，目的是提取content的特征，将节点设置feed_dict为content_pre图片，用新建的一个字典存储这个层的eval
- **前向传播计算style特征**
 - 同样的方法，对每个style_image进行预处理，此时这里style有多个层：relu1_1, relu2_1, relu3_1, relu4_1和relu5_1，对于每个层，feed进入style_image，截取得到从一开始到这几个层的节点，得到其eval的值。每个层都会进行reshape，然后和上一个层的转置相乘，并除以其size，最终和vgg.net一样，每一个层的输出节点都会**存储在dict中**(之后调不调用这些节点要看是否载入这些相应的字典元素)，而不是仅仅得到最终的输出节点。
 - 关于为什么如此操作并不是很理解，但是目的是用前向传播方法获得style的特征
- **利用反向传播，生成风格化图片**
 - 首先设置初始化过后的变量image，用于反向传播进行更新，shape与content_image相同，**下面都image代表生成的风格化后的图片的数据**
 - **计算image和content_image的loss，content_image所采用的节点，是之前在前向传播中设置好的节点，image所采用的节点，是在vgg.net中设置的节点，采用l2_loss算法，其中content有一个weight，用来表示content的风格占比多少**
 - **计算image和style_image的loss，此时style的特征权重是多个层，需要将其相加，求得loss**
 - 总loss之后会加上tv_loss，使其降噪
 - 之后采用反向传播方法，减少image的loss，使其和content以及和style的loss之和减小
 - 进行训练，并在ckpt或者训练结束之时，利用.eval()得到风格化图片的数据，输出
- **补充，由于第二次看这个代码的时候又不理解了，所以补充一下描述：**
 - 首先是输入image的节点，可以直接理解为net，net=vgg.net_preloaded，后面接哪个层就计算到哪个层
 - style_features是style的节点字典
 - 在进行反向传播前，都是只对节点进行存储到字典中，之后可以选择性地运行


#### 总结
**我所理解的风格化的实现过程如下：**
1. content用前向传播提取其特征，style用前向传播提取特征，但是由于content提取的层和style提取的层不同，所以特征的维度不一样，style提取的是低阶特征，具体到形状，而content提取的是高阶特征，对于整幅图。
2. **也就是说，在高阶特征上，风格化的图形尽量和content保持一致，即loss减少，而在低阶特征上，风格化图形尽量和style保持一致，这就表明我们看风格化图片的局部，能够和style图片的局部相似，但是看风格化图片的整体，能够和content图片的整理相似，这是通过控制在哪一层进行loss计算是相关的**
3. **style的特征提取是在relu1_1,relu2_1等较为低阶的层，而content是在relu4_2较为高阶的层，所以在低阶特征上，style有优势，并且这个风格化也是分层的，而content只有一层relu4_2**

#### 展望
- **在这里style图片分别提取了relu1_1,relu2_1等层，可以尝试不同层采用不同的style图片，实现分层的风格化**
- **图片的分层可以采用不同层的relu输出实现低阶到高阶的特征提取，这个例子中style就有很多层特征，而content只有一层高阶特征，利用此实现了风格化，想必应该有很多更加丰富的应用。可以把低阶和高阶的特征分别理解为世界的微观与宏观**



