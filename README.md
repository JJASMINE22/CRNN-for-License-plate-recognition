## 基于CRNN的车牌号识别模型 --Pytorch
---

## 目录  
1. [所需环境 Environment](#所需环境) 
2. [注意事项 Attention](#注意事项) 
3. [网络结构 Network Structure](#网络结构)
4. [效果展示 Effect](#效果展示)
4. [数据下载 Download](#数据下载) 
5. [训练步骤 Train](#训练步骤) 

## 所需环境  
1. Python3.7
2. Pytorch>=1.10.1+cu113  
3. Torchvision >= 0.11.2+cu113
4. Numpy==1.21.6
5. Pillow==8.2.0
6. onnx==1.12.0
7. onnxruntime-gpu==1.8.0
8. CUDA 11.0+
9. Cudnn 8.0.4+

## 注意事项  
1. BatchNorm会导致训练、推理结果存在偏差，将CRNN的BatchNorm替换为LayerNorm，消除批量训练引发的特征偏移
2. 将CRNN的BiLSTM替换为MultiHeadAttention，增强各“词向量”特征
3. 加入正则化操作，降低过拟合影响
4. 数据路径、训练参数、分词内容等均位于./configure目录下
5. onnx通用部署模型转换位于./onnx目录下

## 网络结构
CRNN based on MultiHeadAttention
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1          [-1, 64, 32, 100]           1,792
         MaxPool2d-2           [-1, 64, 16, 50]               0
            Conv2d-3          [-1, 128, 16, 50]          73,856
         MaxPool2d-4           [-1, 128, 8, 25]               0
            Conv2d-5           [-1, 256, 8, 25]         295,168
         LayerNorm-6           [-1, 256, 8, 25]         102,400
            Conv2d-7           [-1, 256, 8, 25]         590,080
         MaxPool2d-8           [-1, 256, 5, 24]               0
            Conv2d-9           [-1, 512, 5, 24]       1,180,160
        LayerNorm-10           [-1, 512, 5, 24]         122,880
           Conv2d-11           [-1, 512, 5, 24]       2,359,808
        MaxPool2d-12           [-1, 512, 3, 23]               0
           Conv2d-13           [-1, 512, 1, 21]       2,359,808
        LayerNorm-14           [-1, 512, 1, 21]          21,504
           Linear-15              [-1, 21, 512]         262,656
           Linear-16              [-1, 21, 512]         262,656
           Linear-17              [-1, 21, 512]         262,656
           Linear-18              [-1, 21, 512]         262,656
        LayerNorm-19              [-1, 21, 512]           1,024
MultiHeadAttention-20  [[-1, 21, 512], [-1, 21, 21]]               0
           Linear-21              [-1, 21, 512]         262,656
           Linear-22              [-1, 21, 512]         262,656
           Linear-23              [-1, 21, 512]         262,656
           Linear-24              [-1, 21, 512]         262,656
        LayerNorm-25              [-1, 21, 512]           1,024
MultiHeadAttention-26  [[-1, 21, 512], [-1, 21, 21]]               0
           Linear-27               [-1, 21, 66]          33,858
================================================================
Total params: 9,244,610
Trainable params: 9,244,610
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.04
Forward/backward pass size (MB): 65.34
Params size (MB): 35.27
Estimated Total Size (MB): 100.65

## 效果展示
![image]()
![image]()
![image]()

## 数据下载    
CCPD2019 
链接：https://github.com/detectRecog/CCPD
下载解压后将数据集放置于config.py中指定的路径。 

## 训练步骤
运行train.py

## 预测步骤
运行predict.py
