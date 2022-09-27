# -*- coding: UTF-8 -*-
'''
@Project ：CRNN
@File    ：networks.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import math
import torch
import torch.nn.functional as F
from torch import nn
from custom.CustomLayers import BidirectionalLSTM, MultiHeadAttention


class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=1, padding_mode='reflect')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=1, padding_mode='reflect')
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=1, padding_mode='reflect')

        self.layer_norm1 = nn.LayerNorm(normalized_shape=[256, 8, 25])

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=1, padding_mode='reflect')
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1),
                                  padding=(1, 0))

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=1, padding_mode='reflect')

        self.layer_norm2 = nn.LayerNorm(normalized_shape=[512, 5, 24])

        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=1, padding_mode='reflect')
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1),
                                  padding=(1, 0))

        self.conv7 = nn.Conv2d(in_channels=512, out_channels=512,
                               kernel_size=(3, 3), stride=(1, 1))

        self.layer_norm3 = nn.LayerNorm(normalized_shape=[512, 1, 21])

        # for attention mode
        self.attn1 = MultiHeadAttention(source_size=512,
                                        embedding_size=512,
                                        multihead_num=4,
                                        drop_rate=0.2)

        self.attn2 = MultiHeadAttention(source_size=512,
                                        embedding_size=512,
                                        multihead_num=4,
                                        drop_rate=0.2)

        self.linear = nn.Linear(in_features=512, out_features=66)

        self.init_params()

    def forward(self, input):
        x = self.conv1(input)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.layer_norm1(x)
        x = F.leaky_relu(x, negative_slope=0.2)

        x = self.conv4(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.pool3(x)

        x = self.conv5(x)
        x = self.layer_norm2(x)
        x = F.leaky_relu(x, negative_slope=0.2)

        x = self.conv6(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.pool4(x)

        x = self.conv7(x)
        x = self.layer_norm3(x)
        x = F.leaky_relu(x, negative_slope=0.2)

        x = x.squeeze(dim=2)
        x = x.permute(0, 2, 1)

        x, _ = self.attn1([x, x, x])
        x, _ = self.attn2([x, x, x])
        x = self.linear(x)

        output = F.log_softmax(x, dim=-1)
        output = output.permute(1, 0, 2)

        return output

    def init_params(self):

        for named_param in self.named_parameters():
            name, param = named_param
            if param.requires_grad:
                if name.split('.')[-1].split('_')[0] == 'weight':
                    stddev = 1 / math.sqrt(param.size(0))
                    torch.nn.init.normal_(param, std=stddev)
                else:
                    torch.nn.init.zeros_(param)

    def get_weights(self):

        weights = []
        for named_param in self.named_parameters():
            name, param = named_param
            if param.requires_grad:
                if name.split('.')[-1].split('_')[0] == 'weight':
                    weights.append(param)
                else:
                    continue
        return weights
