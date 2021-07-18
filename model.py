import torch
from torch import nn
import torch.nn.functional as F

from GlobalAttention import GlobalAttentionGeneral as ATT_NET
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from torchvision import models
import torch.utils.model_zoo as model_zoo
from self_attn import SelfAttention


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])


def conv1x1(in_planes, out_planes, bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=bias)


def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)


# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU())
    return block


# Keep the spatial size
def Block3x3_relu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU())
    return block


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num))

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


# ############## Text2Image Encoder-Decoder #######
class RNN_ENCODER(nn.Module):
    def __init__(self, ntoken, ninput=300, drop_prob=0.5,
                 nhidden=128, nlayers=1, bidirectional=True):
        super(RNN_ENCODER, self).__init__()
        self.n_steps = 10
        self.ntoken = ntoken  # size of the dictionary
        self.ninput = ninput  # size of each embedding vector
        self.drop_prob = drop_prob  # probability of an element to be zeroed
        self.nlayers = nlayers  # Number of recurrent layers
        self.bidirectional = bidirectional
        self.rnn_type = 'LSTM'
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        # number of features in the hidden state
        self.nhidden = nhidden // self.num_directions

        self.define_module()
        self.init_weights()

    def define_module(self):
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == 'LSTM':
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = nn.LSTM(self.ninput, self.nhidden,
                               self.nlayers, batch_first=True,
                               dropout=self.drop_prob,
                               bidirectional=self.bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.ninput, self.nhidden,
                              self.nlayers, batch_first=True,
                              dropout=self.drop_prob,
                              bidirectional=self.bidirectional)
        else:
            raise NotImplementedError

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # Do not need to initialize RNN parameters, which have been initialized
        # http://pytorch.org/docs/master/_modules/torch/nn/modules/rnn.html#LSTM
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.fill_(0)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()),
                    Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()))
        else:
            return Variable(weight.new(self.nlayers * self.num_directions,
                                       bsz, self.nhidden).zero_())

    def forward(self, captions, cap_lens, hidden, mask=None):
        # input: torch.LongTensor of size batch x n_steps
        # --> emb: batch x n_steps x ninput
        emb = self.drop(self.encoder(captions))
        #
        # Returns: a PackedSequence object
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True)
        # #hidden and memory (num_layers * num_directions, batch, hidden_size):
        # tensor containing the initial hidden state for each element in batch.
        # #output (batch, seq_len, hidden_size * num_directions)
        # #or a PackedSequence object:
        # tensor containing output features (h_t) from the last layer of RNN
        output, hidden = self.rnn(emb, hidden)
        # PackedSequence object
        # --> (batch, seq_len, hidden_size * num_directions)
        output = pad_packed_sequence(output, batch_first=True)[0]
        # output = self.drop(output)
        # --> batch x hidden_size*num_directions x seq_len
        words_emb = output.transpose(1, 2)
        # --> batch x num_directions*hidden_size
        if self.rnn_type == 'LSTM':
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)
        return words_emb, sent_emb


class CNN_ENCODER(nn.Module):
    def __init__(self, nef):
        super(CNN_ENCODER, self).__init__()
        self.nef = 256  # define a uniform ranker

        model = models.inception_v3()
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        model.load_state_dict(model_zoo.load_url(url))
        for param in model.parameters():
            param.requires_grad = False
        print('Load pretrained model from ', url)
        # print(model)

        self.define_module(model)
        self.init_trainable_weights()

    def define_module(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c

        self.emb_features = conv1x1(768, self.nef)
        self.emb_cnn_code = nn.Linear(2048, self.nef)

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        features = None
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode='bilinear')(x)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192

        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288

        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768

        # image region features
        features = x
        # 17 x 17 x 768

        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048

        # global image features
        cnn_code = self.emb_cnn_code(x)
        # 512
        if features is not None:
            features = self.emb_features(features)
        return features, cnn_code


class CA_NET(nn.Module):
    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = 256
        self.c_dim = 100
        self.fc = nn.Linear(self.t_dim, self.c_dim * 4, bias=True)
        self.relu = GLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf, ncf):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.in_dim = 100 + ncf
        self.define_module()

    def define_module(self):
        nz, ngf = self.in_dim, self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(nz, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU())

        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)

    def forward(self, z_code, c_code):
        c_z_code = torch.cat((c_code, z_code), 1)
        # state size ngf x 4 x 4
        out_code = self.fc(c_z_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        # state size ngf/3 x 8 x 8
        out_code = self.upsample1(out_code)
        # state size ngf/4 x 16 x 16
        out_code = self.upsample2(out_code)
        # state size ngf/8 x 32 x 32
        out_code32 = self.upsample3(out_code)
        # state size ngf/16 x 64 x 64
        out_code64 = self.upsample4(out_code32)

        return out_code64


class NEXT_STAGE_G(nn.Module):
    def __init__(self, ngf, nef, ncf):
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.ef_dim = nef
        self.cf_dim = ncf
        self.num_residual = 3
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(3):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        self.att = ATT_NET(ngf, self.ef_dim)
        self.residual = self._make_layer(ResBlock, ngf * 2)
        self.upsample = upBlock(ngf * 2, ngf)

    def forward(self, h_code, c_code, word_embs, mask):
        self.att.applyMask(mask)
        c_code, att = self.att(h_code, word_embs)
        h_c_code = torch.cat((h_code, c_code), 1)
        out_code = self.residual(h_c_code)

        out_code = self.upsample(out_code)

        return out_code, att


def get_norm(name, out_channels):
    if name == 'batch':
        norm = nn.BatchNorm2d(out_channels)
    elif name == 'instance':
        norm = nn.InstanceNorm2d(out_channels)
    else:
        norm = None
    return norm


def get_act(name):
    if name == 'relu':
        activation = nn.ReLU(inplace=True)
    elif name == 'elu':
        activation = nn.ELU(inplace=True)
    elif name == 'leaky_relu':
        activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    elif name == 'tanh':
        activation = nn.Tanh()
    elif name == 'sigmoid':
        activation = nn.Sigmoid()
    else:
        activation = None
    return activation


##################### Generator ##########################
class CoarseEncodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 normalization=None, activation=None):
        super().__init__()

        layers = []
        if activation:
            layers.append(get_act(activation))
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1))
        if normalization:
            layers.append(get_norm(normalization, out_channels))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class CoarseDecodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 normalization=None, activation=None):
        super().__init__()

        layers = []
        if activation:
            layers.append(get_act(activation))
        layers.append(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=1))
        if normalization:
            layers.append(get_norm(normalization, out_channels))
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)


class CoarseNet(nn.Module):
    def __init__(self, c_img=3,
                 norm='instance', act_en='leaky_relu', act_de='relu'):
        super().__init__()

        cnum = 64

        self.en_1 = nn.Conv2d(c_img, cnum, 4, 2, padding=1)
        self.en_2 = CoarseEncodeBlock(cnum, cnum * 2, 4, 2, normalization=norm, activation=act_en)
        self.en_3 = CoarseEncodeBlock(cnum * 2, cnum * 4, 4, 2, normalization=norm, activation=act_en)
        self.en_4 = CoarseEncodeBlock(cnum * 4, cnum * 8, 4, 2, normalization=norm, activation=act_en)
        self.en_5 = CoarseEncodeBlock(cnum * 16, cnum * 8, 4, 2, normalization=norm, activation=act_en)
        self.en_6 = CoarseEncodeBlock(cnum * 8, cnum * 8, 4, 2, normalization=norm, activation=act_en)
        self.en_7 = CoarseEncodeBlock(cnum * 8, cnum * 8, 4, 2, normalization=norm, activation=act_en)
        self.en_8 = CoarseEncodeBlock(cnum * 8, cnum * 8, 4, 2, activation=act_en)

        self.de_8 = CoarseDecodeBlock(cnum * 8, cnum * 8, 4, 2, normalization=norm, activation=act_de)
        self.de_7 = CoarseDecodeBlock(cnum * 8 * 2, cnum * 8, 4, 2, normalization=norm, activation=act_de)
        self.de_6 = CoarseDecodeBlock(cnum * 8 * 2, cnum * 8, 4, 2, normalization=norm, activation=act_de)
        self.de_5 = CoarseDecodeBlock(cnum * 8 * 2, cnum * 8, 4, 2, normalization=norm, activation=act_de)
        self.de_4 = CoarseDecodeBlock(cnum * 8 * 2, cnum * 4, 4, 2, normalization=norm, activation=act_de)
        self.de_3 = CoarseDecodeBlock(cnum * 4 * 2, cnum * 2, 4, 2, normalization=norm, activation=act_de)
        self.de_2 = CoarseDecodeBlock(cnum * 2 * 2, cnum, 4, 2, normalization=norm, activation=act_de)
        self.de_1 = nn.Sequential(
            get_act(act_de),
            nn.ConvTranspose2d(cnum * 2, c_img, 4, 2, padding=1),
            get_act('tanh'))

        self.ca_net = CA_NET()
        self.h_net1 = INIT_STAGE_G(512, 100)
        self.h_net2 = NEXT_STAGE_G(32, 256, 100)
        self.h_net3 = NEXT_STAGE_G(32, 256, 100)
        self.attn = SelfAttention(512)

        self.text_en_1 = nn.Conv2d(32, cnum, 4, 2, padding=1)
        self.text_en_2 = CoarseEncodeBlock(cnum, cnum * 2, 4, 2, normalization=norm, activation=act_en)
        self.text_en_3 = CoarseEncodeBlock(cnum * 2, cnum * 4, 4, 2, normalization=norm, activation=act_en)
        self.text_en_4 = CoarseEncodeBlock(cnum * 4, cnum * 8, 4, 2, normalization=norm, activation=act_en)

    def forward(self, x, z_code, sent_emb, word_embs, text_mask):

        ################## Image Encoder 1 ###################
        out_1 = self.en_1(x)
        out_2 = self.en_2(out_1)
        out_3 = self.en_3(out_2)
        out_4 = self.en_4(out_3)
        ##################### Attention ######################
        attn_feature = self.attn(out_4, word_embs)

        ################### Text 2 Image ######################
        att_maps = []
        c_code, mu, logvar = self.ca_net(sent_emb)

        h_code1 = self.h_net1(z_code, c_code)
        h_code2, att1 = self.h_net2(h_code1, c_code, word_embs, text_mask)
        if att1 is not None:
            att_maps.append(att1)
        h_code3, att2 = self.h_net3(h_code2, c_code, attn_feature, text_mask)
        if att2 is not None:
            att_maps.append(att2)

        ################### Text-Image Encoder #################
        text_out_1 = self.text_en_1(h_code3)
        text_out_2 = self.text_en_2(text_out_1)
        text_out_3 = self.text_en_3(text_out_2)
        text_out_4 = self.text_en_4(text_out_3)

        ################## Image Encoder 2 ###################
        out_4_t = torch.cat([out_4, text_out_4], 1)
        out_5 = self.en_5(out_4_t)
        out_6 = self.en_6(out_5)
        out_7 = self.en_7(out_6)
        out_8 = self.en_8(out_7)

        ##################### Decoder #########################
        dout_8 = self.de_8(out_8)
        dout_8_out_7 = torch.cat([dout_8, out_7], 1)
        dout_7 = self.de_7(dout_8_out_7)
        dout_7_out_6 = torch.cat([dout_7, out_6], 1)
        dout_6 = self.de_6(dout_7_out_6)
        dout_6_out_5 = torch.cat([dout_6, out_5], 1)
        dout_5 = self.de_5(dout_6_out_5)
        dout_5_out_4 = torch.cat([dout_5, out_4], 1)
        dout_4 = self.de_4(dout_5_out_4)
        dout_4_out_3 = torch.cat([dout_4, out_3], 1)
        dout_3 = self.de_3(dout_4_out_3)
        dout_3_out_2 = torch.cat([dout_3, out_2], 1)
        dout_2 = self.de_2(dout_3_out_2)
        dout_2_out_1 = torch.cat([dout_2, out_1], 1)
        dout_1 = self.de_1(dout_2_out_1)

        return dout_1, att1, att2, h_code3, c_code


class RefineEncodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 normalization=None, activation=None):
        super().__init__()

        layers = []
        if activation:
            layers.append(get_act(activation))
        layers.append(
            nn.Conv2d(in_channels, in_channels, 4, 2, dilation=2, padding=3))
        if normalization:
            layers.append(get_norm(normalization, out_channels))

        if activation:
            layers.append(get_act(activation))
        layers.append(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=1))
        if normalization:
            layers.append(get_norm(normalization, out_channels))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class RefineDecodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 normalization=None, activation=None):
        super().__init__()

        layers = []
        if activation:
            layers.append(get_act(activation))
        layers.append(
            nn.ConvTranspose2d(in_channels, out_channels, 3, 1, padding=1))
        if normalization:
            layers.append(get_norm(normalization, out_channels))

        if activation:
            layers.append(get_act(activation))
        layers.append(
            nn.ConvTranspose2d(out_channels, out_channels, 4, 2, padding=1))
        if normalization:
            layers.append(get_norm(normalization, out_channels))
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)


class RefineNet(nn.Module):
    def __init__(self, c_img=3,
                 norm='instance', act_en='leaky_relu', act_de='relu'):
        super().__init__()

        c_in = c_img + c_img
        cnum = 64

        self.en_1 = nn.Conv2d(c_in, cnum, 3, 1, padding=1)
        self.en_2 = RefineEncodeBlock(cnum, cnum * 2, normalization=norm, activation=act_en)
        self.en_3 = RefineEncodeBlock(cnum * 2, cnum * 4, normalization=norm, activation=act_en)
        self.en_4 = RefineEncodeBlock(cnum * 4, cnum * 8, normalization=norm, activation=act_en)
        self.en_5 = RefineEncodeBlock(cnum * 8, cnum * 8, normalization=norm, activation=act_en)
        self.en_6 = RefineEncodeBlock(cnum * 16, cnum * 8, normalization=norm, activation=act_en)
        self.en_7 = RefineEncodeBlock(cnum * 8, cnum * 8, normalization=norm, activation=act_en)
        self.en_8 = RefineEncodeBlock(cnum * 8, cnum * 8, normalization=norm, activation=act_en)
        self.en_9 = nn.Sequential(
            get_act(act_en),
            nn.Conv2d(cnum * 8, cnum * 8, 4, 2, padding=1))

        self.de_9 = nn.Sequential(
            get_act(act_de),
            nn.ConvTranspose2d(cnum * 8, cnum * 8, 4, 2, padding=1),
            get_norm(norm, cnum * 8))
        self.de_8 = RefineDecodeBlock(cnum * 8 * 2, cnum * 8, normalization=norm, activation=act_de)
        self.de_7 = RefineDecodeBlock(cnum * 8 * 2, cnum * 8, normalization=norm, activation=act_de)
        self.de_6 = RefineDecodeBlock(cnum * 8 * 2, cnum * 8, normalization=norm, activation=act_de)
        self.de_5 = RefineDecodeBlock(cnum * 8 * 2, cnum * 8, normalization=norm, activation=act_de)
        self.de_4 = RefineDecodeBlock(cnum * 8 * 2, cnum * 4, normalization=norm, activation=act_de)
        self.de_3 = RefineDecodeBlock(cnum * 4 * 2, cnum * 2, normalization=norm, activation=act_de)
        self.de_2 = RefineDecodeBlock(cnum * 2 * 2, cnum, normalization=norm, activation=act_de)
        self.de_1 = nn.Sequential(
            get_act(act_de),
            nn.ConvTranspose2d(cnum * 2, c_img, 3, 1, padding=1))

        self.h_init = nn.Conv2d(32, 32, 4, 2, padding=1)
        self.h_net = NEXT_STAGE_G(32, 256, 100)
        self.attn = SelfAttention(512)

        self.text_en_1 = nn.Conv2d(32, cnum, 4, 2, padding=1)
        self.text_en_2 = RefineEncodeBlock(cnum, cnum * 2, normalization=norm, activation=act_en)
        self.text_en_3 = RefineEncodeBlock(cnum * 2, cnum * 4, normalization=norm, activation=act_en)
        self.text_en_4 = RefineEncodeBlock(cnum * 4, cnum * 8, normalization=norm, activation=act_en)

    def forward(self, x1, x2, h_code, c_code, word_embs, text_mask):
        x = torch.cat([x1, x2], 1)
        out_1 = self.en_1(x)
        out_2 = self.en_2(out_1)
        out_3 = self.en_3(out_2)
        out_4 = self.en_4(out_3)
        out_5 = self.en_5(out_4)
        ##################### Attention ######################
        attn_feature = self.attn(out_5, word_embs)

        ################### Text 2 Image ######################
        h_code_1 = self.h_init(h_code)
        h_code_out, att = self.h_net(h_code_1, c_code, attn_feature, text_mask)

        ################### Text-Image Encoder #################
        text_out_1 = self.text_en_1(h_code_out)
        text_out_2 = self.text_en_2(text_out_1)
        text_out_3 = self.text_en_3(text_out_2)
        text_out_4 = self.text_en_4(text_out_3)

        ################## Image Encoder 2 ###################
        out_5_t = torch.cat([out_5, text_out_4], 1)

        out_6 = self.en_6(out_5_t)
        out_7 = self.en_7(out_6)
        out_8 = self.en_8(out_7)
        out_9 = self.en_9(out_8)

        dout_9 = self.de_9(out_9)
        dout_9_out_8 = torch.cat([dout_9, out_8], 1)
        dout_8 = self.de_8(dout_9_out_8)
        dout_8_out_7 = torch.cat([dout_8, out_7], 1)
        dout_7 = self.de_7(dout_8_out_7)
        dout_7_out_6 = torch.cat([dout_7, out_6], 1)
        dout_6 = self.de_6(dout_7_out_6)
        dout_6_out_5 = torch.cat([dout_6, out_5], 1)
        dout_5 = self.de_5(dout_6_out_5)
        dout_5_out_4 = torch.cat([dout_5, out_4], 1)
        dout_4 = self.de_4(dout_5_out_4)
        dout_4_out_3 = torch.cat([dout_4, out_3], 1)
        dout_3 = self.de_3(dout_4_out_3)
        dout_3_out_2 = torch.cat([dout_3, out_2], 1)
        dout_2 = self.de_2(dout_3_out_2)
        dout_2_out_1 = torch.cat([dout_2, out_1], 1)
        dout_1 = self.de_1(dout_2_out_1)

        return dout_1, att


class CSA(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):
        return x


class InpaintNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.coarse_t = CoarseNet()
        self.refine_t = RefineNet()

        self.weight_conv = nn.Conv2d(16, 1, 3, 1, padding=1)
        self.weight_up = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)

    def forward(self, image, mask, z_code, sent_emb, word_embs, text_mask):
        out_c_t, att1, att2, h_code, c_code = self.coarse_t(image, z_code, sent_emb, word_embs, text_mask)
        out_c_t = image * (1. - mask) + out_c_t * mask

        out_r_t, att_refine_2 = self.refine_t(out_c_t, image, h_code, c_code, word_embs, text_mask)
        out_r_t = image * (1. - mask) + out_r_t * mask

        attn_loss_add = torch.zeros_like(att2, dtype=torch.float)
        attn_loss_add = torch.cat([att2, attn_loss_add, attn_loss_add, attn_loss_add], 1)
        attn_loss_add = attn_loss_add[:, :16, :, :]
        attn_loss = self.weight_conv(attn_loss_add)
        attn_loss = self.weight_up(attn_loss)

        attn_refine_loss_add = torch.zeros_like(att_refine_2, dtype=torch.float)
        attn_refine_loss_add = torch.cat([att_refine_2, attn_refine_loss_add, attn_refine_loss_add, attn_refine_loss_add], 1)
        attn_refine_loss_add = attn_refine_loss_add[:, :16, :, :]
        attn_refine_loss = self.weight_conv(attn_refine_loss_add)
        attn_refine_loss = self.weight_up(attn_refine_loss)

        return out_c_t, out_r_t, attn_loss, attn_refine_loss


class PatchDiscriminator(nn.Module):
    def __init__(self, c_img=3,
                 norm='instance', act='leaky_relu'):
        super().__init__()

        c_in = c_img + c_img
        cnum = 64
        self.discriminator_global = nn.Sequential(
            nn.Conv2d(c_in, cnum, 4, 2, 1),
            get_act(act),

            nn.Conv2d(cnum, cnum * 2, 4, 2, 1),
            get_norm(norm, cnum * 2),
            get_act(act),

            nn.Conv2d(cnum * 2, cnum * 4, 4, 2, 1),
            get_norm(norm, cnum * 4),
            get_act(act),

            nn.Conv2d(cnum * 4, cnum * 8, 4, 1, 1),
            get_norm(norm, cnum * 8),
            get_act(act),

            nn.Conv2d(cnum * 8, 1, 4, 1, 1))

        self.discriminator_local_0 = nn.Sequential(
            nn.Conv2d(c_in, cnum, 5, 1, 2),
            get_act(act),

            nn.Conv2d(cnum, cnum * 2, 4, 2, 1),
            get_norm(norm, cnum * 2),
            get_act(act),

            nn.Conv2d(cnum * 2, cnum * 4, 4, 2, 1),
            get_norm(norm, cnum * 4),
            get_act(act),

            nn.Conv2d(cnum * 4, cnum * 8, 4, 1, 1),
            get_norm(norm, cnum * 8),
            get_act(act),

            nn.Conv2d(cnum * 8, 1, 4, 1, 1))

        self.discriminator_local_1 = nn.Sequential(
            nn.Conv2d(c_in, cnum, 5, 1, 1),
            get_act(act),

            nn.Conv2d(cnum, cnum * 2, 4, 2, 1),
            get_norm(norm, cnum * 2),
            get_act(act),

            nn.Conv2d(cnum * 2, cnum * 4, 4, 2, 1),
            get_norm(norm, cnum * 4),
            get_act(act),

            nn.Conv2d(cnum * 4, cnum * 8, 4, 1, 1),
            get_norm(norm, cnum * 8),
            get_act(act),

            nn.Conv2d(cnum * 8, 1, 4, 1, 1))

        self.discriminator_local_2 = nn.Sequential(
            nn.Conv2d(c_in, cnum, 5, 1, 0),
            get_act(act),

            nn.Conv2d(cnum, cnum * 2, 4, 2, 1),
            get_norm(norm, cnum * 2),
            get_act(act),

            nn.Conv2d(cnum * 2, cnum * 4, 4, 2, 1),
            get_norm(norm, cnum * 4),
            get_act(act),

            nn.Conv2d(cnum * 4, cnum * 8, 4, 1, 1),
            get_norm(norm, cnum * 8),
            get_act(act),

            nn.Conv2d(cnum * 8, 1, 4, 1, 1))

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], 1)
        b, c, h, w = x.shape

        # discriminator_local_0
        # x_local = x[:, :, h//4:h-h//4, w//4:w-w//4]
        # return self.discriminator_global(x), self.discriminator_local_0(x_local)

        # discriminator_local_1
        # x_local = x[:, :, h//4-1:h-h//4+1, w//4-1:w-w//4+1]
        # return self.discriminator_global(x), self.discriminator_local_1(x_local)

        # discriminator_local_2
        x_local = x[:, :, h//4-2:h-h//4+2, w//4-2:w-w//4+2]
        return self.discriminator_global(x), self.discriminator_local_2(x_local)
