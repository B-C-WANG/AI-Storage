# encoding=utf-8

# 本文件提供载入音频文件的函数,提取音频对数幅度谱的函数以及处理文本标签的函数
# 语音的对数频谱作为网络的输入

import torch
import librosa
import torchaudio


def load_audio(path):
    """使用torchaudio读取音频
    Args:
        path(string)            : 音频的路径
    Returns:
        sound(numpy.ndarray)    : 单声道音频数据，如果是多声道进行平均(Samples * 1 channel)
    """
    sound, _ = torchaudio.load(path)
    sound = sound.numpy()
    if len(sound.shape) > 1:
        # 单声道，如果是多声道就平均
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)
    return sound


def parse_audio(path, audio_conf, windows, normalize=False):
    """使用librosa计算音频的对数幅度谱
    Args:
        path(string)       : 音频的路径
        audio_conf(dict)   : 求频谱的参数
        windows(dict)      : 加窗类型
    Returns:
        spect(FloatTensor) : 音频的对数幅度谱(numFrames * nFeatures)
                             nFeatures = n_fft / 2 + 1
    """
    y = load_audio(path)
    n_fft = int(audio_conf['sample_rate'] * audio_conf["window_size"])
    win_length = n_fft
    hop_length = int(audio_conf['sample_rate'] * audio_conf['window_stride'])
    window = windows[audio_conf['window']]
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    spect, phase = librosa.magphase(D)
    spect = torch.FloatTensor(spect)
    spect = spect.log1p()

    # 每句话自己做归一化
    if normalize:
        mean = spect.mean()
        std = spect.std()
        spect.add_(-mean)
        spect.div_(std)
    return spect.transpose(0, 1)


def process_label_file(label_file, char2int):
    """将文本标签处理为数字，转化为numpy类型是为了存储为h5py文件
    Args:
        label_file(string)  :  标签文件路径
        char2int(dict)      :  标签到数字的映射关系 "_'abcdefghijklmnopqrstuvwxyz"
    Output:
        label_dict(list)    :  所有句子的标签，每个句子是list类型
    """
    label_all = []
    with open(label_file, 'r') as f:
        for label in f.readlines():
            label = label.strip()
            label_list = []
            utt = label.split('\t', 1)[0]
            label = label.split('\t', 1)[1]
            for i in range(len(label)):
                if label[i].lower() in char2int:
                    label_list.append(char2int[label[i].lower()])
                else:
                    print("%s not in the label map list" % label[i].lower())
            label_all.append(label_list)
    return label_all


# 本文件继承构建了Dataset类和DataLoader类，用来处理音频和标签文件
# 转化为网络可输入的格式

import os
import torch
import scipy.signal
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils import parse_audio, process_label_file

windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}
audio_conf = {"sample_rate": 16000, 'window_size': 0.025, 'window_stride': 0.01, 'window': 'hamming'}
int2char = ["_", "'", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",
            "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", " "]


class SpeechDataset(Dataset):
    def __init__(self, data_dir, data_set='train', normalize=True):
        self.data_set = data_set
        self.normalize = normalize
        self.char2int = {}
        self.n_feats = int(audio_conf['sample_rate'] * audio_conf['window_size'] / 2 + 1)
        for i in range(len(int2char)):
            self.char2int[int2char[i]] = i

        wav_path = os.path.join(data_dir, data_set + '_wav.scp')
        label_file = os.path.join(data_dir, data_set + '.text')
        self.process_audio(wav_path, label_file)

    def process_audio(self, wav_path, label_file):
        # read the label file
        self.label = process_label_file(label_file, self.char2int)

        # read the path file
        self.path = []
        with open(wav_path, 'r') as f:
            for line in f.readlines():
                utt, path = line.strip().split()
                self.path.append(path)

        # ensure the same samples of input and label
        assert len(self.label) == len(self.path)

    def __getitem__(self, idx):
        return parse_audio(self.path[idx], audio_conf, windows, normalize=self.normalize), self.label[idx]

    def __len__(self):
        return len(self.path)


def collate_fn(batch):
    # 将输入和标签转化为可输入网络的batch
    # batch :     batch_size * (seq_len * nfeats, target_length)
    def func(p):
        return p[0].size(0)

    # sort batch according to the frame nums
    batch = sorted(batch, reverse=True, key=func)
    longest_sample = batch[0][0]
    feat_size = longest_sample.size(1)
    max_length = longest_sample.size(0)
    batch_size = len(batch)

    inputs = torch.zeros(batch_size, max_length, feat_size)  # 网络输入,相当于长度不等的补0
    input_sizes = torch.IntTensor(batch_size)  # 输入每个样本的序列长度，即帧数
    target_sizes = torch.IntTensor(batch_size)  # 每句标签的长度
    targets = []
    input_size_list = []

    for x in range(batch_size):
        sample = batch[x]
        feature = sample[0]
        label = sample[1]
        seq_length = feature.size(0)
        inputs[x].narrow(0, 0, seq_length).copy_(feature)
        input_sizes[x] = seq_length
        input_size_list.append(seq_length)
        target_sizes[x] = len(label)
        targets.extend(label)
    targets = torch.IntTensor(targets)
    return inputs, targets, input_sizes, input_size_list, target_sizes


"""
class torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, 
                                        sampler=None, batch_sampler=None, num_workers=0, 
                                        collate_fn=<function default_collate>, 
                                        pin_memory=False, drop_last=False)
"""


class SpeechDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(SpeechDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = collate_fn


# encoding=utf-8

# greedy decoder and beamsearch decoder for ctc

import torch


class Decoder(object):
    "解码器基类定义，作用是将模型的输出转化为文本使其能够与标签计算正确率"

    def __init__(self, int2char, space_idx=28, blank_index=0):
        """
        int2char     :     将类别转化为字符标签
        space_idx    :     空格符号的索引，如果为为-1，表示空格不是一个类别
        blank_index  :     空白类的索引，默认设置为0
        """
        self.int_to_char = int2char
        self.space_idx = space_idx
        self.blank_index = blank_index
        self.num_word = 0
        self.num_char = 0

    def decode(self):
        "解码函数，在GreedyDecoder和BeamDecoder继承类中实现"
        raise NotImplementedError

    def phone_word_error(self, prob_tensor, frame_seq_len, targets, target_sizes):
        """计算词错率和字符错误率
        Args:
            prob_tensor     :   模型的输出
            frame_seq_len   :   每个样本的帧长
            targets         :   样本标签
            target_sizes    :   每个样本标签的长度
        Returns:
            wer             :   词错率，以space为间隔分开作为词
            cer             :   字符错误率
        """
        strings = self.decode(prob_tensor, frame_seq_len)
        targets = self._unflatten_targets(targets, target_sizes)
        target_strings = self._process_strings(self._convert_to_strings(targets))

        cer = 0
        wer = 0
        for x in range(len(target_strings)):
            cer += self.cer(strings[x], target_strings[x])
            wer += self.wer(strings[x], target_strings[x])
            self.num_word += len(target_strings[x].split())
            self.num_char += len(target_strings[x])
        return cer, wer

    def _unflatten_targets(self, targets, target_sizes):
        """将标签按照每个样本的标签长度进行分割
        Args:
            targets        :    数字表示的标签
            target_sizes   :    每个样本标签的长度
        Returns:
            split_targets  :    得到的分割后的标签
        """
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset: offset + size])
            offset += size
        return split_targets

    def _process_strings(self, seqs, remove_rep=False):
        """处理转化后的字符序列，包括去重复等，将list转化为string
        Args:
            seqs       :   待处理序列
            remove_rep :   是否去重复
        Returns:
            processed_strings  :  处理后的字符序列
        """
        processed_strings = []
        for seq in seqs:
            string = self._process_string(seq, remove_rep)
            processed_strings.append(string)
        return processed_strings

    def _process_string(self, seq, remove_rep=False):
        string = ''
        for i, char in enumerate(seq):
            if char != self.int_to_char[self.blank_index]:
                if remove_rep and i != 0 and char == seq[i - 1]:  # remove dumplicates
                    pass
                elif self.space_idx == -1:
                    string = string + ' ' + char
                elif char == self.int_to_char[self.space_idx]:
                    string += ' '
                else:
                    string = string + char
        return string

    def _convert_to_strings(self, seq, sizes=None):
        """将数字序列的输出转化为字符序列
        Args:
            seqs       :   待转化序列
            sizes      :   每个样本序列的长度
        Returns:
            strings  :  转化后的字符序列
        """
        strings = []
        for x in range(len(seq)):
            seq_len = sizes[x] if sizes is not None else len(seq[x])
            string = self._convert_to_string(seq[x], seq_len)
            strings.append(string)
        return strings

    def _convert_to_string(self, seq, sizes):
        result = []
        for i in range(sizes):
            result.append(self.int_to_char[seq[i]])
        if self.space_idx == -1:
            return result
        else:
            return ''.join(result)

    def wer(self, s1, s2):
        "将空格作为分割计算词错误率"
        b = set(s1.split() + s2.split())
        word2int = dict(zip(b, range(len(b))))

        w1 = [word2int[w] for w in s1.split()]
        w2 = [word2int[w] for w in s2.split()]
        return self._edit_distance(w1, w2)

    def cer(self, s1, s2):
        "计算字符错误率"
        return self._edit_distance(s1, s2)

    def _edit_distance(self, src_seq, tgt_seq):
        "计算两个序列的编辑距离，用来计算字符错误率"
        L1, L2 = len(src_seq), len(tgt_seq)
        if L1 == 0: return L2
        if L2 == 0: return L1
        # construct matrix of size (L1 + 1, L2 + 1)
        dist = [[0] * (L2 + 1) for i in range(L1 + 1)]
        for i in range(1, L2 + 1):
            dist[0][i] = dist[0][i - 1] + 1
        for i in range(1, L1 + 1):
            dist[i][0] = dist[i - 1][0] + 1
        for i in range(1, L1 + 1):
            for j in range(1, L2 + 1):
                if src_seq[i - 1] == tgt_seq[j - 1]:
                    cost = 0
                else:
                    cost = 1
                dist[i][j] = min(dist[i][j - 1] + 1, dist[i - 1][j] + 1, dist[i - 1][j - 1] + cost)
        return dist[L1][L2]


class GreedyDecoder(Decoder):
    "直接解码，把每一帧的输出概率最大的值作为输出值，而不是整个序列概率最大的值"

    def decode(self, prob_tensor, frame_seq_len):
        """解码函数
        Args:
            prob_tensor   :   网络模型输出
            frame_seq_len :   每一样本的帧数
        Returns:
            解码得到的string，即识别结果
        """
        prob_tensor = prob_tensor.transpose(0, 1)
        _, decoded = torch.max(prob_tensor, 2)
        decoded = decoded.view(decoded.size(0), decoded.size(1))
        decoded = self._convert_to_strings(decoded, frame_seq_len)  # convert digit idx to chars
        return self._process_strings(decoded, remove_rep=True)


class BeamDecoder(Decoder):
    "Beam search 解码。解码结果为整个序列概率的最大值"

    def __init__(self, int2char, beam_width=200, blank_index=0, space_idx=28):
        self.beam_width = beam_width
        super(BeamDecoder, self).__init__(int2char, space_idx=space_idx, blank_index=blank_index)

        import BeamSearch
        self._decoder = BeamSearch.ctcBeamSearch(int2char, beam_width, blank_index=blank_index)

    def decode(self, prob_tensor, frame_seq_len=None):
        """解码函数
        Args:
            prob_tensor   :   网络模型输出
            frame_seq_len :   每一样本的帧数
        Returns:
            res           :   解码得到的string，即识别结果
        """
        probs = prob_tensor.transpose(0, 1)
        res = self._decoder.decode(probs, frame_seq_len)
        return res


# encoding=utf-8

# 模型文件

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class SequenceWise(nn.Module):
    """调整输入满足module的需求，因为多次使用，所以模块化构建一个类
    适用于将LSTM的输出通过batchnorm或者Linear层
    """

    def __init__(self, module):
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        """
        Args:
            x :    PackedSequence
        """
        x, batch_size_len = x.data, x.batch_sizes
        # x.data:    sum(x_len) * num_features
        x = self.module(x)
        x = nn.utils.rnn.PackedSequence(x, batch_size_len)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class BatchSoftmax(nn.Module):
    """
    The layer to add softmax for a sequence, which is the output of rnn
    Which state use its own softmax, and concat the result
    """

    def forward(self, x):
        # x: seq_len * batch_size * num
        if not self.training:
            seq_len = x.size()[0]
            return torch.stack([F.softmax(x[i], dim=1) for i in range(seq_len)], 0)
        else:
            return x


class BatchRNN(nn.Module):
    """
    Add BatchNorm before rnn to generate a batchrnn layer
    """

    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM,
                 bidirectional=False, batch_norm=True, dropout=0.1):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, dropout=dropout, bias=False)

    def forward(self, x):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x, _ = self.rnn(x)
        # self.rnn.flatten_parameters()
        return x


class CTC_Model(nn.Module):
    def __init__(self, rnn_param=None, num_class=48, drop_out=0.1):
        """
        rnn_param(dict)  :  the dict of rnn parameters
                            rnn_param = {"rnn_input_size":201, "rnn_hidden_size":256, ....}
        num_class(int)   :  the number of units, add one for blank to be the classes to classify
        drop_out(float)  :  drop_out paramteter for all place where need drop_out
        """
        super(CTC_Model, self).__init__()
        if rnn_param is None or type(rnn_param) != dict:
            raise ValueError("rnn_param need to be a dict to contain all params of rnn!")
        self.rnn_param = rnn_param
        self.num_class = num_class
        self.num_directions = 2 if rnn_param["bidirectional"] else 1
        self._drop_out = drop_out

        rnn_input_size = rnn_param["rnn_input_size"]
        rnns = []

        rnn_hidden_size = rnn_param["rnn_hidden_size"]
        rnn_type = rnn_param["rnn_type"]
        rnn_layers = rnn_param["rnn_layers"]
        bidirectional = rnn_param["bidirectional"]
        batch_norm = rnn_param["batch_norm"]

        rnn = BatchRNN(input_size=rnn_input_size, hidden_size=rnn_hidden_size,
                       rnn_type=rnn_type, bidirectional=bidirectional, dropout=drop_out,
                       batch_norm=False)

        rnns.append(('0', rnn))
        # 堆叠RNN,除了第一次不使用batchnorm，其他层RNN都加入BachNorm
        for i in range(rnn_layers - 1):
            rnn = BatchRNN(input_size=self.num_directions * rnn_hidden_size,
                           hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                           bidirectional=bidirectional, dropout=drop_out, batch_norm=batch_norm)
            rnns.append(('%d' % (i + 1), rnn))

        self.rnns = nn.Sequential(OrderedDict(rnns))

        if batch_norm:
            fc = nn.Sequential(nn.BatchNorm1d(self.num_directions * rnn_hidden_size),
                               nn.Linear(self.num_directions * rnn_hidden_size, num_class + 1, bias=False), )
        else:
            fc = nn.Linear(self.num_directions * rnn_hidden_size, num_class + 1, bias=False)

        self.fc = SequenceWise(fc)
        self.inference_softmax = BatchSoftmax()

    def forward(self, x, dev=False):
        x = self.rnns(x)
        x = self.fc(x)
        x, batch_seq = nn.utils.rnn.pad_packed_sequence(x, batch_first=False)

        out = self.inference_softmax(x)
        if dev:
            return x, out  # 如果是验证集，需要同时返回x计算loss和out进行wer的计算
        return out

    @staticmethod
    def save_package(model, optimizer=None, decoder=None, epoch=None, loss_results=None, dev_loss_results=None,
                     dev_cer_results=None):
        package = {
            'rnn_param': model.rnn_param,
            'num_class': model.num_class,
            '_drop_out': model._drop_out,
            'state_dict': model.state_dict()
        }
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()
        if decoder is not None:
            package['decoder'] = decoder
        if epoch is not None:
            package['epoch'] = epoch
        if loss_results is not None:
            package['loss_results'] = loss_results
            package['dev_loss_results'] = dev_loss_results
            package['dev_cer_results'] = dev_cer_results
        return package


# encoding=utf-8

# 本文件为训练训练声学模型文件

import os
import sys
import time
import copy
import torch
import argparse
import numpy as np

try:
    import ConfigParser
except:
    import configparser as ConfigParser
import torch.nn as nn
from torch.autograd import Variable

from model import *
from decoder import GreedyDecoder
from warpctc_pytorch import CTCLoss
from data import int2char, SpeechDataset, SpeechDataLoader

# 支持的rnn类型
RNN = {'lstm': nn.LSTM, 'rnn': nn.RNN, 'gru': nn.GRU}

parser = argparse.ArgumentParser(description='lstm_ctc')
parser.add_argument('--conf', default='./conf/ctc_model_setting.conf',
                    help='conf file with Argument of LSTM and training')


def train(model, train_loader, loss_fn, optimizer, logger, print_every=20, USE_CUDA=True):
    """训练一个epoch，即将整个训练集跑一次
    Args:
        model         :  定义的网络模型
        train_loader  :  加载训练集的类对象
        loss_fn       :  损失函数，此处为CTCLoss
        optimizer     :  优化器类对象
        logger        :  日志类对象
        print_every   :  每20个batch打印一次loss
        USE_CUDA      :  是否使用GPU
    Returns:
        average_loss  :  一个epoch的平均loss
    """
    model.train()

    total_loss = 0
    print_loss = 0
    i = 0
    for data in train_loader:
        inputs, targets, input_sizes, input_sizes_list, target_sizes = data
        batch_size = inputs.size(0)
        inputs = inputs.transpose(0, 1)

        inputs = Variable(inputs, requires_grad=False)
        input_sizes = Variable(input_sizes, requires_grad=False)
        targets = Variable(targets, requires_grad=False)
        target_sizes = Variable(target_sizes, requires_grad=False)

        if USE_CUDA:
            inputs = inputs.cuda()

        inputs = nn.utils.rnn.pack_padded_sequence(inputs, input_sizes_list)

        out = model(inputs)
        loss = loss_fn(out, targets, input_sizes, target_sizes)
        loss /= batch_size
        print_loss += loss.data[0]

        if (i + 1) % print_every == 0:
            print('batch = %d, loss = %.4f' % (i + 1, print_loss / print_every))
            logger.debug('batch = %d, loss = %.4f' % (i + 1, print_loss / print_every))
            print_loss = 0

        total_loss += loss.data[0]
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), 400)
        optimizer.step()
        i += 1
    average_loss = total_loss / i
    print("Epoch done, average loss: %.4f" % average_loss)
    logger.info("Epoch done, average loss: %.4f" % average_loss)
    return average_loss


def dev(model, dev_loader, loss_fn, decoder, logger, USE_CUDA=True):
    """验证集的计算过程，与train()不同的是不需要反向传播过程，并且需要计算字符正确率
    Args:
        model       :   模型
        dev_loader  :   加载验证集的类对象
        loss_fn     :   损失函数
        decoder     :   解码类对象，即将网络的输出解码成文本
        logger      :   日志类对象
        USE_CUDA    :   是否使用GPU
    Returns:
        acc * 100    :   字符正确率，如果space不是一个标签的话，则为词正确率
        average_loss :   验证集的平均loss
    """
    model.eval()
    total_cer = 0
    total_tokens = 0
    total_loss = 0
    i = 0

    for data in dev_loader:
        inputs, targets, input_sizes, input_sizes_list, target_sizes = data
        batch_size = inputs.size(0)
        inputs = inputs.transpose(0, 1)

        inputs = Variable(inputs, requires_grad=False)
        input_sizes = Variable(input_sizes, requires_grad=False)
        targets = Variable(targets, requires_grad=False)
        target_sizes = Variable(target_sizes, requires_grad=False)

        if USE_CUDA:
            inputs = inputs.cuda()

        inputs = nn.utils.rnn.pack_padded_sequence(inputs, input_sizes_list)
        out, probs = model(inputs, dev=True)

        loss = loss_fn(out, targets, input_sizes, target_sizes)
        loss /= batch_size
        total_loss += loss.data[0]

        probs = probs.data.cpu()
        targets = targets.data
        target_sizes = target_sizes.data

        if decoder.space_idx == -1:
            total_cer += decoder.phone_word_error(probs, input_sizes_list, targets, target_sizes)[1]
        else:
            total_cer += decoder.phone_word_error(probs, input_sizes_list, targets, target_sizes)[0]
        total_tokens += sum(target_sizes)
        i += 1
    acc = 1 - float(total_cer) / total_tokens
    average_loss = total_loss / i
    return acc * 100, average_loss


def init_logger(log_file):
    """得到一个日志的类对象
    Args:
        log_file   :  日志文件名
    Returns:
        logger     :  日志类对象
    """
    import logging
    from logging.handlers import RotatingFileHandler

    logger = logging.getLogger()
    hdl = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=10)
    formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
    hdl.setFormatter(formatter)
    logger.addHandler(hdl)
    logger.setLevel(logging.DEBUG)
    return logger


def main():
    args = parser.parse_args()
    cf = ConfigParser.ConfigParser()
    try:
        cf.read(args.conf)
    except:
        print("conf file not exists")
        sys.exit(1)
    USE_CUDA = cf.getboolean('Training', 'use_cuda')
    try:
        seed = long(cf.get('Training', 'seed'))
    except:
        seed = torch.cuda.initial_seed()
        cf.set('Training', 'seed', seed)
        cf.write(open(args.conf, 'w'))

    torch.manual_seed(seed)
    if USE_CUDA:
        torch.cuda.manual_seed(seed)

    log_dir = cf.get('Data', 'log_dir')
    log_file = os.path.join(log_dir, cf.get('Data', 'log_file'))
    logger = init_logger(log_file)

    # Define Model
    rnn_input_size = cf.getint('Model', 'rnn_input_size')
    rnn_hidden_size = cf.getint('Model', 'rnn_hidden_size')
    rnn_layers = cf.getint('Model', 'rnn_layers')
    rnn_type = RNN[cf.get('Model', 'rnn_type')]
    bidirectional = cf.getboolean('Model', 'bidirectional')
    batch_norm = cf.getboolean('Model', 'batch_norm')
    rnn_param = {"rnn_input_size": rnn_input_size, "rnn_hidden_size": rnn_hidden_size, "rnn_layers": rnn_layers,
                 "rnn_type": rnn_type, "bidirectional": bidirectional, "batch_norm": batch_norm}
    num_class = cf.getint('Model', 'num_class')
    drop_out = cf.getfloat('Model', 'drop_out')

    model = CTC_Model(rnn_param=rnn_param, num_class=num_class, drop_out=drop_out)
    print("Model Structure:")
    logger.info("Model Structure:")
    for idx, m in enumerate(model.children()):
        print(idx, m)
        logger.info(str(idx) + "->" + str(m))

    data_dir = cf.get('Data', 'data_dir')
    batch_size = cf.getint("Training", 'batch_size')

    # Data Loader
    train_dataset = SpeechDataset(data_dir, data_set='train')
    dev_dataset = SpeechDataset(data_dir, data_set="dev")
    train_loader = SpeechDataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                    num_workers=4, pin_memory=False)
    dev_loader = SpeechDataLoader(dev_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=4, pin_memory=False)

    # ensure the feats is equal to the rnn_input_Size
    assert train_dataset.n_feats == rnn_input_size

    # decoder for dev set
    decoder = GreedyDecoder(int2char, space_idx=len(int2char) - 1, blank_index=0)

    # Training
    init_lr = cf.getfloat('Training', 'init_lr')
    num_epoches = cf.getint('Training', 'num_epoches')
    end_adjust_acc = cf.getfloat('Training', 'end_adjust_acc')
    decay = cf.getfloat("Training", 'lr_decay')
    weight_decay = cf.getfloat("Training", 'weight_decay')

    params = {'num_epoches': num_epoches, 'end_adjust_acc': end_adjust_acc, 'seed': seed,
              'decay': decay, 'learning_rate': init_lr, 'weight_decay': weight_decay, 'batch_size': batch_size,
              'n_feats': train_dataset.n_feats}
    print(params)

    if USE_CUDA:
        model = model.cuda()

    loss_fn = CTCLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay)

    # visualization for training
    from visdom import Visdom
    viz = Visdom()
    title = 'TIMIT LSTM_CTC Acoustic Model'

    opts = [dict(title=title + " Loss", ylabel='Loss', xlabel='Epoch'),
            dict(title=title + " Loss on Dev", ylabel='DEV Loss', xlabel='Epoch'),
            dict(title=title + ' CER on DEV', ylabel='DEV CER', xlabel='Epoch')]
    viz_window = [None, None, None]

    count = 0
    learning_rate = init_lr
    loss_best = 1000
    loss_best_true = 1000
    adjust_rate_flag = False
    stop_train = False
    adjust_time = 0
    acc_best = 0
    start_time = time.time()
    loss_results = []
    dev_loss_results = []
    dev_cer_results = []

    while not stop_train:
        if count >= num_epoches:
            break
        count += 1

        if adjust_rate_flag:
            learning_rate *= decay
            adjust_rate_flag = False
            for param in optimizer.param_groups:
                param['lr'] *= decay

        print("Start training epoch: %d, learning_rate: %.5f" % (count, learning_rate))
        logger.info("Start training epoch: %d, learning_rate: %.5f" % (count, learning_rate))

        loss = train(model, train_loader, loss_fn, optimizer, logger, print_every=20, USE_CUDA=USE_CUDA)
        loss_results.append(loss)
        acc, dev_loss = dev(model, dev_loader, loss_fn, decoder, logger, USE_CUDA=USE_CUDA)
        print("loss on dev set is %.4f" % dev_loss)
        logger.info("loss on dev set is %.4f" % dev_loss)
        dev_loss_results.append(dev_loss)
        dev_cer_results.append(acc)

        # adjust learning rate by dev_loss
        # adjust_rate_count  :  表示连续超过count个epoch的loss在end_adjust_acc区间内认为稳定
        if dev_loss < (loss_best - end_adjust_acc):
            loss_best = dev_loss
            loss_best_true = dev_loss
            adjust_rate_count = 0
            acc_best = acc
            best_model_state = copy.deepcopy(model.state_dict())
            best_op_state = copy.deepcopy(optimizer.state_dict())
        elif (dev_loss < loss_best + end_adjust_acc):
            adjust_rate_count += 1
            if dev_loss < loss_best and dev_loss < loss_best_true:
                loss_best_true = dev_loss
                acc_best = acc
                best_model_state = copy.deepcopy(model.state_dict())
                best_op_state = copy.deepcopy(optimizer.state_dict())
        else:
            adjust_rate_count = 10

        print("adjust_rate_count: %d" % adjust_rate_count)
        print('adjust_time: %d' % adjust_time)
        logger.info("adjust_rate_count: %d" % adjust_rate_count)
        logger.info('adjust_time: %d' % adjust_time)

        if adjust_rate_count == 10:
            adjust_rate_flag = True
            adjust_time += 1
            adjust_rate_count = 0
            if loss_best > loss_best_true:
                loss_best = loss_best_true
            model.load_state_dict(best_model_state)
            optimizer.load_state_dict(best_op_state)

        if adjust_time == 8:
            stop_train = True

        time_used = (time.time() - start_time) / 60
        print("epoch %d done, dev acc is: %.4f, time_used: %.4f minutes" % (count, acc, time_used))
        logger.info("epoch %d done, dev acc is: %.4f, time_used: %.4f minutes" % (count, acc, time_used))

        x_axis = range(count)
        y_axis = [loss_results[0:count], dev_loss_results[0:count], dev_cer_results[0:count]]
        for x in range(len(viz_window)):
            if viz_window[x] is None:
                viz_window[x] = viz.line(X=np.array(x_axis), Y=np.array(y_axis[x]), opts=opts[x], )
            else:
                viz.line(X=np.array(x_axis), Y=np.array(y_axis[x]), win=viz_window[x], update='replace', )

    print("End training, best dev loss is: %.4f, acc is: %.4f" % (loss_best_true, acc_best))
    logger.info("End training, best dev loss acc is: %.4f, acc is: %.4f" % (loss_best_true, acc_best))
    model.load_state_dict(best_model_state)
    optimizer.load_state_dict(best_op_state)
    best_path = os.path.join(log_dir, 'best_model' + '_dev' + str(acc_best) + '.pkl')
    cf.set('Model', 'model_file', best_path)
    cf.write(open(args.conf, 'w'))
    params['epoch'] = count

    torch.save(CTC_Model.save_package(model, optimizer=optimizer, epoch=params, loss_results=loss_results,
                                      dev_loss_results=dev_loss_results, dev_cer_results=dev_cer_results), best_path)


if __name__ == '__main__':
    main()
