# -*- coding: UTF-8 -*-
'''
@Project ：CRNN
@File    ：CustomLayers.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class BidirectionalLSTM(nn.Module):
    def __init__(self,
                 input_size:int=None,
                 hidden_size:int=None,
                 target_size:int=None,
                 **kwargs):
        super(BidirectionalLSTM, self).__init__(**kwargs)
        self.target_size = target_size
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            batch_first=True,
                            bidirectional=True)
        self.linear = nn.Linear(in_features=2*hidden_size,
                                out_features=target_size)

    def forward(self, input):

        x, _ = self.lstm(input)
        output = self.linear(x)

        return output


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 source_size: int=None,
                 embedding_size: int=None,
                 multihead_num: int=None,
                 drop_rate: float=None):
        super(MultiHeadAttention, self).__init__()
        self.source_size = source_size
        self.embedding_size = embedding_size
        self.multihead_num = multihead_num
        self.drop_rate = drop_rate

        self.linear = nn.Linear(in_features=self.embedding_size, out_features=self.source_size)
        self.linear_q = nn.Linear(in_features=self.source_size, out_features=self.embedding_size)
        self.linear_k = nn.Linear(in_features=self.source_size, out_features=self.embedding_size)
        self.linear_v = nn.Linear(in_features=self.source_size, out_features=self.embedding_size)

        self.layer_norm = nn.LayerNorm(normalized_shape=self.source_size)

    def forward(self, inputs, mask=None):

        assert isinstance(inputs, list)
        q = inputs[0]
        k = inputs[1]
        v = inputs[-1] if len(inputs) == 3 else k
        batch_size = q.size(0)

        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        # note: attr split_size_or_sections indicates the size of the slice, not the num of the slice
        q = torch.cat(torch.split(q, split_size_or_sections=self.embedding_size//self.multihead_num, dim=-1), dim=0)
        k = torch.cat(torch.split(k, split_size_or_sections=self.embedding_size//self.multihead_num, dim=-1), dim=0)
        v = torch.cat(torch.split(v, split_size_or_sections=self.embedding_size//self.multihead_num, dim=-1), dim=0)

        attention = torch.matmul(q, k.transpose(2, 1))/torch.sqrt(torch.tensor(self.embedding_size
                                                                               //self.multihead_num, dtype=torch.float))

        if mask is not None:
            # mask = mask.repeat(self.multihead_num, 1, 1)
            attention -= 1e+9 * mask
        attention = torch.softmax(attention, dim=-1)

        feature = torch.matmul(attention, v)
        feature = torch.cat(torch.split(feature, split_size_or_sections=batch_size, dim=0), dim=-1)
        output = self.linear(feature)
        output = torch.dropout(output, p=self.drop_rate, train=self.training)

        output = torch.add(output, inputs[0])

        output = self.layer_norm(output)

        return output, attention

class ResidueBlock(nn.Module):
    def __init__(self,
                 short_cut: bool,
                 drop_rate: float,
                 in_channels: int,
                 out_channels: int,
                 padding_mode: str,
                 stride: int or tuple,
                 negative_slope: float,
                 kernel_size: int or tuple,
                 padding: str or tuple or int):
        super(ResidueBlock, self).__init__()
        self.stride = stride
        self.padding = padding
        self.drop_rate = drop_rate
        self.short_cut = short_cut
        self.drop_rate = drop_rate
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding_mode = padding_mode
        self.negative_slope = negative_slope

        if np.not_equal(self.in_channels, self.out_channels):
            self.conv1 = nn.Conv2d(in_channels=self.in_channels,
                                   out_channels=self.out_channels,
                                   kernel_size=self.kernel_size,
                                   stride=self.stride,
                                   padding=self.padding,
                                   padding_mode=self.padding_mode)

        self.conv2 = nn.Conv2d(in_channels=self.in_channels,
                               out_channels=self.out_channels//2 if self.short_cut else self.out_channels,
                               kernel_size=self.kernel_size,
                               stride=self.stride,
                               padding=self.padding,
                               padding_mode=self.padding_mode)
        self.bn1 = nn.BatchNorm2d(num_features=self.out_channels//2 if self.short_cut else self.out_channels)

        self.conv3 = nn.Conv2d(in_channels=self.out_channels//2 if self.short_cut else self.out_channels,
                               out_channels=self.out_channels,
                               kernel_size=self.kernel_size,
                               stride=(1, 1),
                               padding='same',
                               padding_mode=self.padding_mode)
        self.bn2 = nn.BatchNorm2d(num_features=self.out_channels)

    def forward(self, init):

        x = init
        if np.not_equal(self.in_channels, self.out_channels):
            init = self.conv1(init)

        x = self.conv2(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=self.negative_slope)

        x = self.conv3(x)
        x = torch.dropout(x, p=self.drop_rate, train=True)

        x = torch.add(x, init)
        x = self.bn2(x)

        output = F.leaky_relu(x, negative_slope=self.negative_slope)

        return output

class SeparableConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 padding_mode: str,
                 depthwise_stride: int or tuple,
                 depthwise_ksize: int or tuple,
                 padding: str or tuple or int):
        super(SeparableConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depthwise_padding = padding
        self.depthwise_stride = depthwise_stride
        self.depthwise_ksize = depthwise_ksize
        self.pointwise_stride = (1, 1)
        self.pointwise_ksize = (1, 1)
        self.padding_mode = padding_mode

        self.depthwise_conv = nn.Conv2d(in_channels=self.in_channels,
                                        out_channels=self.in_channels,
                                        kernel_size=self.depthwise_ksize,
                                        stride=self.depthwise_stride,
                                        padding=self.depthwise_padding,
                                        padding_mode=self.padding_mode,
                                        groups=self.in_channels)

        self.pointwise_cov = nn.Conv2d(in_channels=self.in_channels,
                                       out_channels=self.out_channels,
                                       kernel_size=self.pointwise_ksize,
                                       stride=self.pointwise_stride)

    def forward(self, input):

        x = self.depthwise_conv(input)

        output = self.pointwise_cov(x)

        return output


class MobileBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 padding_mode: str,
                 depthwise_stride: int,
                 padding: str or tuple or int,
                 expansion_factor: float):
        super(MobileBlock, self).__init__()
        assert padding in [0, (0, 0), 'valid']
        self.depthwise_ksize = (3, 3)
        self.pointwise_stride = (1, 1)
        self.pointwise_ksize = (1, 1)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding_mode = padding_mode
        self.depthwise_padding = padding
        self.depthwise_stride = depthwise_stride
        self.expansion_factor = expansion_factor

        self.depthwise_conv = nn.Conv2d(in_channels=int(self.in_channels*self.expansion_factor),
                                        out_channels=int(self.in_channels*self.expansion_factor),
                                        kernel_size=self.depthwise_ksize,
                                        stride=(self.depthwise_stride,)*2,
                                        padding=self.depthwise_padding,
                                        padding_mode=self.padding_mode,
                                        groups=int(self.in_channels*self.expansion_factor))

        self.pointwise_conv1 = nn.Conv2d(in_channels=self.in_channels,
                                         out_channels=int(self.in_channels*self.expansion_factor),
                                         kernel_size=self.pointwise_ksize,
                                         stride=self.pointwise_stride)

        self.pointwise_conv2 = nn.Conv2d(in_channels=int(self.in_channels*self.expansion_factor),
                                         out_channels=self.out_channels,
                                         kernel_size=self.pointwise_ksize,
                                         stride=self.pointwise_stride)

        self.bn1 = nn.BatchNorm2d(num_features=int(self.in_channels*self.expansion_factor))

        self.bn2 = nn.BatchNorm2d(num_features=int(self.in_channels*self.expansion_factor))

        self.bn3 = nn.BatchNorm2d(num_features=self.out_channels)

    def forward(self, init):
        x = init

        x = self.pointwise_conv1(x)
        x = self.bn1(x)
        x = F.relu6(x)

        if np.greater(self.depthwise_stride, 1):
            x = F.pad(x, pad=(1,)*4, mode=self.padding_mode)

        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = F.relu6(x)

        x = self.pointwise_conv2(x)

        if np.logical_and(np.equal(self.in_channels, self.out_channels),
                          np.equal(self.depthwise_stride, 1)):
            x = torch.add(x, init)

        output = self.bn3(x)

        return output
