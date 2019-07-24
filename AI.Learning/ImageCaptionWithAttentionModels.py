import torch
from torch import nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size
        # 读取预训练模型的resnet层，带上预先训练的权重
        resnet = torchvision.models.resnet101(pretrained=True)
        # 去掉最后两层，提取出特征层
        modules = list(resnet.children())[:-2]
        # 重新构成网络
        self.resnet = nn.Sequential(*modules)
        # 自适应池化Adaptive Pooling会将所有输入自适应池化成固定大小的输出，使用平均池化，保证输出固定
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        # 对fine tuning进行必要的设置，具体进入函数查看
        self.fine_tune()

    def forward(self, images):
        """
        输入Image
        返回最终的feature map特征
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        # 拿出最终的特征层
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        # fine tune先将最后几层变为可以优化，允许梯度，
        # 然后进行fine tuning分类任务，会留下最后两层Dense和Softmax
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        #  encoder的第三维度到attention，注意第三个维度是通道，第二个维度是长宽像素点个数
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        #  decoder的上一个结果，向量
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        # attention到softmax层，注意这里是到1，是通道变为1，每个像素点有一个softmax结果，于是就有了一个和图片相同大小的概率通道
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        # softmax layer to calculate weights
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        # 从encoder到attention，可以看成每个像素从encoder_dim的通道到了attention_dim的通道
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        # decoder都是一维向量
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        # 这里将图片的通道的attention结果和上一个decoder层的attention结果进行相加，relu得到softmax概率结果
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        # 得到的softmax的大小是和图片相同大小，相当于1个通道的图片
        # 再和encoder结果乘在一起，相当于图片的所有通道都乘上了一个概率，体现了attention，作用回X，可以按照残差网络理解，
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    大的Decoder层，带上Attention

    主要的LSTM层：
    显式输入：输入图片encoder结果+上一个预测出的词的embedding
    隐藏权重输入：除了第一次输入图片encoder out结果（通过self.init_hidden_state(encoder_out)）以外
                其他情况下都是使用上一次输出的h
    显式输出：输出decoder向量，softmax到vocab大小的向量，得到输出各个词的概率，可以看成词的onehot，通过argmax得到词

    Attention的体现：
    图片的encoder结果的各个通道经过Dense变为attention通道，然后和上一次LSTM decoder输出的结果Dense过后的attention向量相加
    然后softmax到1，对于每个像素点有一个softmax结果，然后把这个概率结果和encoder out相乘，得到新的
    encoder out后放入LSTM中

    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        # 从图片encoder out，（encoder dim是图片的通道大小），和上一个LSTM输出的decoder，经过attention dim的Dense然后softmax
        # 最终将softmax乘以encoder dim给LSTM
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network
        # embedding层，多少个词，每个词用多少个向量
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        # LSTM输入采用embedding层的特征加上encoder层（这个encoder是attention过后的）
        # 输出的decoder一方面会用于softmax到词的onehot，另一方面会用于attention计算encoder out的每个像素的概率
        # 注意这里没有涉及到LSTM内部的h权重的传播，因为h传播写在LSTM内部，有且仅有init_hidden_state的时候会对h赋值，其他情况下都是上一次的h传到下一次
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.
        载入预训练的词向量w2v
        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        # 使用decoder的结果来初始化LSTM里面的hidden权重，只需要第一次初始化，之后都是完全内部传播
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state，这里需要注意，LSTM层内部的h一开始使用encoder输出初始化，过后的h就是LSTM上一层的输出
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
