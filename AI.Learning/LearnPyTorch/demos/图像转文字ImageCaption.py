# coding:utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch as t
from torch.utils import data
import os
from PIL import Image
import torchvision as tv
import numpy as np

'''
image caption 原理：
image通过CNN提取语义信息f
将f输入LSTM，期望输出S0（输出的是softmax结果，每个词的概率）
S0输入LSTM期望输出S1
这样直到输出Sn
S0到Sn就是n个单词，组成一句话

图片经过ResNet提取2048维度向量，然后全连接成256维度
相当于图像语义空间转到词向量语义空间
经过embedding，每个词变成了256的向量
然后把图片的256向量和embedding的词向量concat
然后LSTM
LSTM之后进行分类，把LSTM特征转到每个词的概率中
（相当于image然后concat前n-1个词，然后预测后n-1个词语）

具体来讲：
提取图像特征得到256向量v0，通过LSTM得到输出w1，对应找概率最大的词语
将w1这个词语通过embedding得到词向量v1，然后输入LSTM得到下一个
然后依次将下一个词语一直这样，直到遇到结束标志符

** 注意：LSTM第一次输入的是图片的feature，之后输入的就是LSTM上一次输出的结果了，因为LSTM有记忆功能，所以这样做
没有问题，能够将图片的信息传递下去！！所以归根结底，还是feature的传递尤为重要，CNN只不过是特征提取，RNN也是序列特征向量的处理手段！

但是这样是贪心的算法，因为每次找的是概率最大的词语

于是可以使用beam search，每次搜索找最可能的k个单词


'''

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# - 区分训练集和验证集
# - 不是随机返回每句话，而是根据index%5
# -

# def create_collate_fn():
#     def collate_fn():
#         pass
#     return collate_fn

def create_collate_fn(padding, eos, max_length=50):
    def collate_fn(img_cap):
        """
        将多个样本拼接在一起成一个batch
        输入： list of data，形如
        [(img1, cap1, index1), (img2, cap2, index2) ....]

        拼接策略如下：
        - batch中每个样本的描述长度都是在变化的，不丢弃任何一个词\
          选取长度最长的句子，将所有句子pad成一样长
        - 长度不够的用</PAD>在结尾PAD
        - 没有START标识符
        - 如果长度刚好和词一样，那么就没有</EOS>

        返回：
        - imgs(Tensor): batch_size*2048
        - cap_tensor(Tensor): batch_size*max_length
        - lengths(list of int): 长度为batch_size
        - index(list of int): 长度为batch_size
        """
        img_cap.sort(key=lambda p: len(p[1]), reverse=True)
        imgs, caps, indexs = zip(*img_cap)
        imgs = t.cat([img.unsqueeze(0) for img in imgs], 0)
        lengths = [min(len(c) + 1, max_length) for c in caps]
        batch_length = max(lengths)
        cap_tensor = t.LongTensor(batch_length, len(caps)).fill_(padding)
        for i, c in enumerate(caps):
            end_cap = lengths[i] - 1
            if end_cap < batch_length:
                cap_tensor[end_cap, i] = eos
            cap_tensor[:end_cap, i].copy_(c[:end_cap])
        return (imgs, (cap_tensor, lengths), indexs)

    return collate_fn


class CaptionDataset(data.Dataset):

    def __init__(self, opt):
        """
        Attributes:
            _data (dict): 预处理之后的数据，包括所有图片的文件名，以及处理过后的描述
            all_imgs (tensor): 利用resnet50提取的图片特征，形状（200000，2048）
            caption(list): 长度为20万的list，包括每张图片的文字描述
            ix2id(dict): 指定序号的图片对应的文件名
            start_(int): 起始序号，训练集的起始序号是0，验证集的起始序号是190000，即
                前190000张图片是训练集，剩下的10000张图片是验证集
            len_(init): 数据集大小，如果是训练集，长度就是190000，验证集长度为10000
            traininig(bool): 是训练集(True),还是验证集(False)

            相当于从图片特征到文字的特征


        """
        self.opt = opt
        data = t.load(opt.caption_data_path)
        word2ix = data['word2ix']
        self.captions = data['caption']
        self.padding = word2ix.get(data.get('padding'))
        self.end = word2ix.get(data.get('end'))
        self._data = data
        self.ix2id = data['ix2id']
        self.all_imgs = t.load(opt.img_feature_path)

    def __getitem__(self, index):
        """
        返回：
        - img: 图像features 2048的向量
        - caption: 描述，形如LongTensor([1,3,5,2]),长度取决于描述长度
        - index: 下标，图像的序号，可以通过ix2id[index]获取对应图片文件名
        """
        img = self.all_imgs[index]

        caption = self.captions[index]
        # 5句描述随机选一句
        rdn_index = np.random.choice(len(caption), 1)[0]
        caption = caption[rdn_index]
        return img, t.LongTensor(caption), index

    def __len__(self):
        return len(self.ix2id)


def get_dataloader(opt):
    dataset = CaptionDataset(opt)
    dataloader = data.DataLoader(dataset,
                                 batch_size=opt.batch_size,
                                 shuffle=opt.shuffle,
                                 num_workers=opt.num_workers,
                                 collate_fn=create_collate_fn(dataset.padding, dataset.end))
    return dataloader




import torch as t
import numpy as np
import json
import jieba
import tqdm


class Config:
    annotation_file = 'caption_train_annotations_20170902.json'
    unknown = '</UNKNOWN>'
    end = '</EOS>'
    padding = '</PAD>'
    max_words = 10000
    min_appear = 2
    save_path = 'caption.pth'


# START='</START>'
# MAX_LENS = 25,

def process(**kwargs):
    opt = Config()
    for k, v in kwargs.items():
        setattr(opt, k, v)

    with open(opt.annotation_file) as f:
        data = json.load(f)

    # 8f00f3d0f1008e085ab660e70dffced16a8259f6.jpg -> 0
    id2ix = {item['image_id']: ix for ix, item in enumerate(data)}
    # 0-> 8f00f3d0f1008e085ab660e70dffced16a8259f6.jpg
    ix2id = {ix: id for id, ix in (id2ix.items())}
    assert id2ix[ix2id[10]] == 10

    captions = [item['caption'] for item in data]
    # 分词结果
    cut_captions = [[list(jieba.cut(ii, cut_all=False)) for ii in item] for item in tqdm.tqdm(captions)]

    word_nums = {}  # '快乐'-> 10000 (次)

    def update(word_nums):
        def fun(word):
            word_nums[word] = word_nums.get(word, 0) + 1
            return None

        return fun

    lambda_ = update(word_nums)
    _ = {lambda_(word) for sentences in cut_captions for sentence in sentences for word in sentence}

    # [ (10000,u'快乐')，(9999,u'开心') ...]
    word_nums_list = sorted([(num, word) for word, num in word_nums.items()], reverse=True)

    #### 以上的操作是无损，可逆的操作###############################
    # **********以下会删除一些信息******************

    # 1. 丢弃词频不够的词
    # 2. ~~丢弃长度过长的词~~

    words = [word[1] for word in word_nums_list[:opt.max_words] if word[0] >= opt.min_appear]
    words = [opt.unknown, opt.padding, opt.end] + words
    word2ix = {word: ix for ix, word in enumerate(words)}
    ix2word = {ix: word for word, ix in word2ix.items()}
    assert word2ix[ix2word[123]] == 123

    ix_captions = [[[word2ix.get(word, word2ix.get(opt.unknown)) for word in sentence]
                    for sentence in item]
                   for item in cut_captions]
    readme = u"""
    word：词
    ix:index
    id:图片名
    caption: 分词之后的描述，通过ix2word可以获得原始中文词
    """
    results = {
        'caption': ix_captions,
        'word2ix': word2ix,
        'ix2word': ix2word,
        'ix2id': ix2id,
        'id2ix': id2ix,
        'padding': '</PAD>',
        'end': '</EOS>',
        'readme': readme
    }
    t.save(results, opt.save_path)
    print('save file in %s' % opt.save_path)

    def test(ix, ix2=4):
        results = t.load(opt.save_path)
        ix2word = results['ix2word']
        examples = results['caption'][ix][4]
        sentences_p = (''.join([ix2word[ii] for ii in examples]))
        sentences_r = data[ix]['caption'][ix2]
        assert sentences_p == sentences_r, 'test failed'

    test(1000)
    print('test success')



# coding:utf8
"""
利用resnet50提取图片的语义信息
并保存层results.pth
"""
from config import Config
import tqdm
import torch as t
from torch.autograd import Variable
import torchvision as tv
from torch.utils import data
import os
from PIL import Image
import numpy as np

t.set_grad_enabled(False)
opt = Config()

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
normalize = tv.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


class CaptionDataset(data.Dataset):

    def __init__(self, caption_data_path):
        self.transforms = tv.transforms.Compose([
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(256),
            tv.transforms.ToTensor(),
            normalize
        ])

        data = t.load(caption_data_path)
        self.ix2id = data['ix2id']
        # 所有图片的路径
        self.imgs = [os.path.join(opt.img_path, self.ix2id[_]) \
                     for _ in range(len(self.ix2id))]

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB')
        img = self.transforms(img)
        return img, index

    def __len__(self):
        return len(self.imgs)


def get_dataloader(opt):
    dataset = CaptionDataset(opt.caption_data_path)
    dataloader = data.DataLoader(dataset,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 num_workers=opt.num_workers,
                                 )
    return dataloader


# 数据
opt.batch_size = 256
dataloader = get_dataloader(opt)
results = t.Tensor(len(dataloader.dataset), 2048).fill_(0)
batch_size = opt.batch_size

# 模型
resnet50 = tv.models.resnet50(pretrained=True)
del resnet50.fc
resnet50.fc = lambda x: x
resnet50.cuda()

# 前向传播，计算分数
for ii, (imgs, indexs) in tqdm.tqdm(enumerate(dataloader)):
    # 确保序号没有对应错
    assert indexs[0] == batch_size * ii
    imgs = imgs.cuda()
    features = resnet50(imgs)
    results[ii * batch_size:(ii + 1) * batch_size] = features.data.cpu()

# 200000*2048 20万张图片，每张图片2048维的feature
t.save(results, 'results.pth')

#coding:utf8
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Class for generating captions from an image-to-text model.
Adapted from https://github.com/tensorflow/models/blob/master/im2txt/im2txt/inference_utils/caption_generator.py"""



import torch
from torch.autograd import Variable
from torch.nn.functional import log_softmax
import heapq


class Caption(object):
    """Represents a complete or partial caption."""

    def __init__(self, sentence, state, logprob, score, metadata=None):
        """Initializes the Caption.
        Args:
          sentence: List of word ids in the caption.
          state: Model state after generating the previous word.
          logprob: Log-probability of the caption.
          score: Score of the caption.
          metadata: Optional metadata associated with the partial sentence. If not
            None, a list of strings with the same length as 'sentence'.
        """
        self.sentence = sentence
        self.state = state
        self.logprob = logprob
        self.score = score
        self.metadata = metadata

    def __cmp__(self, other):
        """Compares Captions by score."""
        assert isinstance(other, Caption)
        if self.score == other.score:
            return 0
        elif self.score < other.score:
            return -1
        else:
            return 1

    # For Python 3 compatibility (__cmp__ is deprecated).
    def __lt__(self, other):
        assert isinstance(other, Caption)
        return self.score < other.score

    # Also for Python 3 compatibility.
    def __eq__(self, other):
        assert isinstance(other, Caption)
        return self.score == other.score


class TopN(object):
    """Maintains the top n elements of an incrementally provided set."""

    def __init__(self, n):
        self._n = n
        self._data = []

    def size(self):
        assert self._data is not None
        return len(self._data)

    def push(self, x):
        """Pushes a new element."""
        assert self._data is not None
        if len(self._data) < self._n:
            heapq.heappush(self._data, x)
        else:
            heapq.heappushpop(self._data, x)

    def extract(self, sort=False):
        """Extracts all elements from the TopN. This is a destructive operation.
        The only method that can be called immediately after extract() is reset().
        Args:
          sort: Whether to return the elements in descending sorted order.
        Returns:
          A list of data; the top n elements provided to the set.
        """
        assert self._data is not None
        data = self._data
        self._data = None
        if sort:
            data.sort(reverse=True)
        return data

    def reset(self):
        """Returns the TopN to an empty state."""
        self._data = []


class CaptionGenerator(object):
    """Class to generate captions from an image-to-text model."""

    def __init__(self,
                 embedder,
                 rnn,
                 classifier,
                 eos_id,
                 beam_size=3,
                 max_caption_length=20,
                 length_normalization_factor=0.0):
        """Initializes the generator.
        Args:
          model: recurrent model, with inputs: (input, state) and outputs len(vocab) values
          beam_size: Beam size to use when generating captions.
          max_caption_length: The maximum caption length before stopping the search.
          length_normalization_factor: If != 0, a number x such that captions are
            scored by logprob/length^x, rather than logprob. This changes the
            relative scores of captions depending on their lengths. For example, if
            x > 0 then longer captions will be favored.
        """
        self.embedder = embedder
        self.rnn = rnn
        self.classifier = classifier
        self.eos_id = eos_id
        self.beam_size = beam_size
        self.max_caption_length = max_caption_length
        self.length_normalization_factor = length_normalization_factor

    def beam_search(self, rnn_input, initial_state=None):
        """Runs beam search caption generation on a single image.
        Args:
          initial_state: An initial state for the recurrent model
        Returns:
          A list of Caption sorted by descending score.
        """

        def get_topk_words(embeddings, state):
            output, new_states = self.rnn(embeddings, state)
            output = self.classifier(output.squeeze(0))
            logprobs = log_softmax(output, dim=1)
            logprobs, words = logprobs.topk(self.beam_size, 1)
            return words.data, logprobs.data, new_states

        partial_captions = TopN(self.beam_size)
        complete_captions = TopN(self.beam_size)

        words, logprobs, new_state = get_topk_words(rnn_input, initial_state)
        for k in range(self.beam_size):
            cap = Caption(
                sentence=[words[0, k]],
                state=new_state,
                logprob=logprobs[0, k],
                score=logprobs[0, k])
            partial_captions.push(cap)

        # Run beam search.
        for _ in range(self.max_caption_length - 1):
            partial_captions_list = partial_captions.extract()
            partial_captions.reset()
            input_feed = torch.LongTensor([c.sentence[-1]
                                           for c in partial_captions_list])
            if rnn_input.is_cuda:
                input_feed = input_feed.cuda()
            input_feed = Variable(input_feed, volatile=True)
            state_feed = [c.state for c in partial_captions_list]
            if isinstance(state_feed[0], tuple):
                state_feed_h, state_feed_c = zip(*state_feed)
                state_feed = (torch.cat(state_feed_h, 1),
                              torch.cat(state_feed_c, 1))
            else:
                state_feed = torch.cat(state_feed, 1)

            embeddings = self.embedder(input_feed).view(1, len(input_feed), -1)
            words, logprobs, new_states = get_topk_words(
                embeddings, state_feed)
            for i, partial_caption in enumerate(partial_captions_list):
                if isinstance(new_states, tuple):
                    state = (new_states[0].narrow(1, i, 1),
                             new_states[1].narrow(1, i, 1))
                else:
                    state = new_states[i]
                for k in range(self.beam_size):
                    w = words[i, k]
                    sentence = partial_caption.sentence + [w]
                    logprob = partial_caption.logprob + logprobs[i, k]
                    score = logprob
                    if w == self.eos_id:
                        if self.length_normalization_factor > 0:
                            score /= len(sentence)**self.length_normalization_factor
                        beam = Caption(sentence, state, logprob, score)
                        complete_captions.push(beam)
                    else:
                        beam = Caption(sentence, state, logprob, score)
                        partial_captions.push(beam)
            if partial_captions.size() == 0:
                # We have run out of partial candidates; happens when beam_size
                # = 1.
                break

        # If we have no complete captions then fall back to the partial captions.
        # But never output a mixture of complete and partial captions because a
        # partial caption could have a higher score than all the complete
        # captions.
        if not complete_captions.size():
            complete_captions = partial_captions

        caps = complete_captions.extract(sort=True)

        return [c.sentence for c in caps], [c.score for c in caps]


# coding:utf8
import torch as t
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

import time


class CaptionModel(nn.Module):
    def __init__(self, opt, word2ix, ix2word):
        super(CaptionModel, self).__init__()
        self.ix2word = ix2word
        self.word2ix = word2ix
        self.opt = opt

        self.fc = nn.Linear(2048, opt.rnn_hidden)

        self.rnn = nn.LSTM(opt.embedding_dim, opt.rnn_hidden, num_layers=opt.num_layers)
        self.classifier = nn.Linear(opt.rnn_hidden, len(word2ix))
        self.embedding = nn.Embedding(len(word2ix), opt.embedding_dim)
        # if opt.share_embedding_weights:
        #     # rnn_hidden=embedding_dim的时候才可以
        #     self.embedding.weight

    def forward(self, img_feats, captions, lengths):
        # 输入是image的ResNet特征向量，以及用index表示的句子，或者说是单词的index的list
        embeddings = self.embedding(captions)
        # img_feats是2048维的向量,通过全连接层转为256维的向量,和词向量一样
        img_feats = self.fc(img_feats).unsqueeze(0)
        # 将img_feats看成第一个词的词向量
        embeddings = t.cat([img_feats, embeddings], 0)
        # PackedSequence
        packed_embeddings = pack_padded_sequence(embeddings, lengths)
        # LSTM输入和输出都是packed sequence，所以这里要转换
        outputs, state = self.rnn(packed_embeddings)
        # lstm的输出作为特征用来分类预测下一个词的序号
        # 因为输入是PackedSequence,所以输出的output也是PackedSequence

        # TODO：这里是不是把语句变成了batch的数据，然后训练batch？？

        # PackedSequence第一个元素是Variable,第二个元素是batch_sizes,
        # 即batch中每个样本的长度

        # 这里是一个多分类任务！

        pred = self.classifier(outputs[0])
        return pred, state

    def generate(self, img, eos_token='</EOS>',
                 beam_size=3,
                 max_caption_length=30,
                 length_normalization_factor=0.0):
        """
        根据图片生成描述,主要是使用beam search算法以得到更好的描述
        """
        cap_gen = CaptionGenerator(embedder=self.embedding,
                                   rnn=self.rnn,
                                   classifier=self.classifier,
                                   eos_id=self.word2ix[eos_token],
                                   beam_size=beam_size,
                                   max_caption_length=max_caption_length,
                                   length_normalization_factor=length_normalization_factor)
        if next(self.parameters()).is_cuda:
            img = img.cuda()
        img =img.unsqueeze(0)
        img = self.fc(img).unsqueeze(0)
        sentences, score = cap_gen.beam_search(img)
        sentences = [' '.join([self.ix2word[idx] for idx in sent])
                     for sent in sentences]
        return sentences

    def states(self):
        opt_state_dict = {attr: getattr(self.opt, attr)
                          for attr in dir(self.opt)
                          if not attr.startswith('__')}
        return {
            'state_dict': self.state_dict(),
            'opt': opt_state_dict
        }

    def save(self, path=None, **kwargs):
        if path is None:
            path = '{prefix}_{time}'.format(prefix=self.opt.prefix,
                                            time=time.strftime('%m%d_%H%M'))
        states = self.states()
        states.update(kwargs)
        t.save(states, path)
        return path

    def load(self, path, load_opt=False):
        data = t.load(path, map_location=lambda s, l: s)
        state_dict = data['state_dict']
        self.load_state_dict(state_dict)

        if load_opt:
            for k, v in data['opt'].items():
                setattr(self.opt, k, v)

        return self

    def get_optimizer(self, lr):
        return t.optim.Adam(self.parameters(), lr=lr)


# coding:utf8
import os
import torch as t
import torchvision as tv
from torchnet import meter
import tqdm

from torch.nn.utils.rnn import pack_padded_sequence

from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def generate(**kwargs):
    opt = Config()
    for k, v in kwargs.items():
        setattr(opt, k, v)
    device = t.device('cuda') if opt.use_gpu else t.device('cpu')

    # 数据预处理
    data = t.load(opt.caption_data_path, map_location=lambda s, l: s)
    word2ix, ix2word = data['word2ix'], data['ix2word']

    normalize = tv.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    transforms = tv.transforms.Compose([
        tv.transforms.Resize(opt.scale_size),
        tv.transforms.CenterCrop(opt.img_size),
        tv.transforms.ToTensor(),
        normalize
    ])
    img = Image.open(opt.test_img)
    img = transforms(img).unsqueeze(0)

    # 用resnet50来提取图片特征
    resnet50 = tv.models.resnet50(True).eval()
    del resnet50.fc
    resnet50.fc = lambda x: x
    resnet50.to(device)
    img = img.to(device)
    img_feats = resnet50(img).detach()

    # Caption模型
    model = CaptionModel(opt, word2ix, ix2word)
    model = model.load(opt.model_ckpt).eval()
    model.to(device)

    results = model.generate(img_feats.data[0])
    print('\r\n'.join(results))


def train(**kwargs):
    opt = Config()
    for k, v in kwargs.items():
        setattr(opt, k, v)
    device = t.device('cuda') if opt.use_gpu else t.device('cpu')

    opt.caption_data_path = 'caption.pth'  # 原始数据
    opt.test_img = ''  # 输入图片
    # opt.model_ckpt='caption_0914_1947' # 预训练的模型

    # 数据
    vis = Visualizer(env=opt.env)
    dataloader = get_dataloader(opt)
    _data = dataloader.dataset._data
    word2ix, ix2word = _data['word2ix'], _data['ix2word']

    # 模型
    model = CaptionModel(opt, word2ix, ix2word)
    if opt.model_ckpt:
        model.load(opt.model_ckpt)
    optimizer = model.get_optimizer(opt.lr)
    criterion = t.nn.CrossEntropyLoss()

    model.to(device)

    # 统计
    loss_meter = meter.AverageValueMeter()

    for epoch in range(opt.epoch):
        loss_meter.reset()
        for ii, (imgs, (captions, lengths), indexes) in tqdm.tqdm(enumerate(dataloader)):
            # 训练
            optimizer.zero_grad()
            imgs = imgs.to(device)
            captions = captions.to(device)
            input_captions = captions[:-1]
            target_captions = pack_padded_sequence(captions, lengths)[0]
            score, _ = model(imgs, input_captions, lengths)
            loss = criterion(score, target_captions)
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.item())

            # 可视化
            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                vis.plot('loss', loss_meter.value()[0])

                # 可视化原始图片 + 可视化人工的描述语句
                raw_img = _data['ix2id'][indexes[0]]
                img_path = opt.img_path + raw_img
                raw_img = Image.open(img_path).convert('RGB')
                raw_img = tv.transforms.ToTensor()(raw_img)

                raw_caption = captions.data[:, 0]
                raw_caption = ''.join([_data['ix2word'][ii] for ii in raw_caption])
                vis.text(raw_caption, u'raw_caption')
                vis.img('raw', raw_img, caption=raw_caption)

                # 可视化网络生成的描述语句
                results = model.generate(imgs.data[0])
                vis.text('</br>'.join(results), u'caption')
        model.save()
