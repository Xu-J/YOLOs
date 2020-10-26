import torch.nn as nn
import torch.nn.functional as F
# import torchsummary as summary
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        '''第一层卷积层，卷积核为3*3，通道数为96，步距为1，原始图像大小为32*32，有R、G、B三个通道'''

        '''这样经过第一层卷积层之后，得到的feature map的大小为(32-3)/1+1=30,所以feature map的维度为96*30*30'''

        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=1)

        '''经过一次批归一化，将数据拉回到正态分布'''

        self.bn1 = nn.BatchNorm2d(96)

        '''第一层池化层，卷积核为3*3，步距为2，前一层的feature map的大小为30*30，通道数为96个'''

        '''这样经过第一层池化层之后，得到的feature map的大小为(30-3)/2+1=14,所以feature map的维度为96*14*14'''

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        '''第二层卷积层，卷积核为3*3，通道数为256，步距为1，前一层的feature map的大小为14*14，通道数为96个'''

        '''这样经过第一层卷积层之后，得到的feature map的大小为(14-3)/1+1=12,所以feature map的维度为256*12*12'''

        self.conv2 = nn.Conv2d(96, 256, kernel_size=3, stride=1)

        '''经过一次批归一化，将数据拉回到正态分布'''

        self.bn2 = nn.BatchNorm2d(256)

        '''第二层池化层，卷积核为3*3，步距为2，前一层的feature map的大小为12*12，通道数为256个'''

        '''这样经过第二层池化层之后，得到的feature map的大小为(12-3)/2+1=5,所以feature map的维度为256*5*5'''

        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        '''第三层卷积层，卷积核为3*3，通道数为384，步距为1，前一层的feature map的大小为5*5，通道数为256个'''

        '''这样经过第一层卷积层之后，得到的feature map的大小为(5-3+2*1)/1+1=5,所以feature map的维度为384*5*5'''

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1, stride=1)

        '''第四层卷积层，卷积核为3*3，通道数为384，步距为1，前一层的feature map的大小为5*5，通道数为384个'''

        '''这样经过第一层卷积层之后，得到的feature map的大小为(5-3+2*1)/1+1=5,所以feature map的维度为384*5*5'''

        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1, stride=1)

        '''第五层卷积层，卷积核为3*3，通道数为384，步距为1，前一层的feature map的大小为5*5，通道数为384个'''

        '''这样经过第一层卷积层之后，得到的feature map的大小为(5-3+2*1)/1+1=5,所以feature map的维度为256*5*5'''

        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1, stride=1)

        '''第三层池化层，卷积核为3*3，步距为2，前一层的feature map的大小为5*5，通道数为256个'''

        '''这样经过第三层池化层之后，得到的feature map的大小为(5-3)/2+1=2,所以feature map的维度为256*2*2'''

        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        '''经过第一层全连接层'''

        self.linear1 = nn.Linear(1024, 2048)

        '''经过第一次DropOut层'''

        self.dropout1 = nn.Dropout(0.5)

        '''经过第二层全连接层'''

        self.linear2 = nn.Linear(2048, 2048)

        '''经过第二层DropOut层'''

        self.dropout2 = nn.Dropout(0.5)

        '''经过第三层全连接层，得到输出结果'''

        self.linear3 = nn.Linear(2048, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.pool2(out)

        out = F.relu(self.conv3(out))

        out = F.relu(self.conv4(out))

        out = F.relu(self.conv5(out))

        out = self.pool3(out)

        out = out.reshape(-1, 256 * 2 * 2) #打平

        out = F.relu(self.linear1(out))

        out = self.dropout1(out)

        out = F.relu(self.linear2(out))

        out = self.dropout2(out)

        out = self.linear3(out)

        return out


model = AlexNet()
print(model)
# summary.summary(model, input_size=(3, 32, 32), batch_size=128, device="cpu")

# class AlexNet(nn.Module):
#     def __init__(self, num_classes=1000):  # imagenet数量
#         super().__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             LRN(local_size=5, alpha=1e-4, beta=0.75, ACROSS_CHANNELS=True)
#         )
#
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, groups=2, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             LRN(local_size=5, alpha=1e-4, beta=0.75, ACROSS_CHANNELS=True)
#         )
#
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(in_channels=256, out_channels=384, padding=1, kernel_size=3),
#             nn.ReLU(inplace=True)
#         )
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )
#
#         self.layer5 = nn.Sequential(
#             nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2)
#         )
#
#         # 需要针对上一层改变view
#         self.layer6 = nn.Sequential(
#             nn.Linear(in_features=6 * 6 * 256, out_features=4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout()
#         )
#         self.layer7 = nn.Sequential(
#             nn.Linear(in_features=4096, out_features=4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout()
#         )
#
#         self.layer8 = nn.Linear(in_features=4096, out_features=num_classes)
#
#     def forward(self, x):
#         x = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
#         x = x.view(-1, 6 * 6 * 256)
#         x = self.layer8(self.layer7(self.layer6(x)))
#
#         return x


# -*- coding: utf-8 -*-
# create the yolov1 net

__author = 'Lizzie'

import torch
import torch.nn as nn
import math


class Net(nn.Module):

    def __init__(self):
        '''
        super(),__init__() calls uper classes' __init__()
        '''
        super(Net, self).__init__()
        self.img_size = 448  # build cfg please
        # self.classes = cfg.classes	# build cfg please
        '''
        nn.Conv2d(input_channels,out_channels,kernel_size,stride,padding,...)
        padding = x, put x to four sides of input
        example: input.size() = (1,1,4,4); padding = 1; after_padding.size() = (1,1,6,6)
        nn.MaxPool2d(kernel_size, stride, padding,...)
        '''
        '''
        P1
        '''
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2,
                               padding=3)  # output of conv1 (size) = (batch_size, 64, 224, 224)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)  # output size = (batch_size, 64, 112, 112)
        '''
        P2
        '''
        self.conv2 = nn.Conv2d(64, 192, 3, 1, 1)  # (b_s, 192, 112, 112)
        self.mp2 = nn.MaxPool2d(2, 2)  # (b_s, 192, 56, 56)
        '''
        P3
        '''
        self.conv3 = nn.Conv2d(192, 128, 1, 1, 0)  # (b_s, 128, 56, 56)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)  # (b_s, 256, 56, 56)
        self.conv5 = nn.Conv2d(256, 256, 1, 1, 0)  # (b_s, 256, 56, 56)
        self.conv6 = nn.Conv2d(256, 512, 3, 1, 1)  # (b_s, 512, 56, 56)
        self.mp3 = nn.MaxPool2d(2, 2)  # (b_s, 512, 28, 28)
        '''
        P4
        '''
        self.conv7 = nn.Conv2d(512, 256, 1, 1, 0)  # (b_s, 256, 28, 28)	rd=1
        self.conv8 = nn.Conv2d(256, 512, 3, 1, 1)  # (b_s, 512, 28, 28)	rd=1
        self.conv9 = nn.Conv2d(512, 256, 1, 1, 0)  # (b_s, 256, 28, 28)	rd=2
        self.conv10 = nn.Conv2d(256, 512, 3, 1, 1)  # (b_s, 512, 28, 28)	rd=2
        self.conv11 = nn.Conv2d(512, 256, 1, 1, 0)  # (b_s, 256, 28, 28)	rd=3
        self.conv12 = nn.Conv2d(256, 512, 3, 1, 1)  # (b_s, 512, 28, 28)	rd=3
        self.conv13 = nn.Conv2d(512, 256, 1, 1, 0)  # (b_s, 256, 28, 28)	rd=4
        self.conv14 = nn.Conv2d(256, 512, 3, 1, 1)  # (b_s, 512, 28, 28)	rd=4
        self.conv15 = nn.Conv2d(512, 512, 1, 1, 0)  # (b_s, 512, 28, 28)
        self.conv16 = nn.Conv2d(512, 1024, 3, 1, 1)  # (b_s, 1024, 28, 28)
        self.mp4 = nn.MaxPool2d(2, 2)  # (b_s, 1024, 14, 14)
        '''
        P5
        '''
        self.conv17 = nn.Conv2d(1024, 512, 1, 1, 0)  # (b_s, 512, 14, 14)	rd=1
        self.conv18 = nn.Conv2d(512, 1024, 3, 1, 1)  # (b_s, 1024, 14, 14)	rd=1
        self.conv19 = nn.Conv2d(1024, 512, 1, 1, 0)  # (b_s, 512, 14, 14)	rd=2
        self.conv20 = nn.Conv2d(512, 1024, 3, 1, 1)  # (b_s, 1024, 14, 14)	rd=2
        self.conv21 = nn.Conv2d(1024, 1024, 1, 1, 0)  # (b_s, 1024, 14, 14)
        self.conv22 = nn.Conv2d(1024, 1024, 3, 2, 1)  # (b_s, 1024, 7, 7)
        '''
        P6
        '''
        self.conv23 = nn.Conv2d(1024, 1024, 3, 1, 1)  # (b_s, 1024, 7, 7)
        self.conv24 = nn.Conv2d(1024, 1024, 3, 1, 1)  # (b_s, 1024, 7, 7)
        '''
        P7
        '''
        self.fc1 = nn.Linear(1024 * 7 * 7, 4096)
        '''
        P8
        '''
        self.fc2 = nn.Linear(4096, 30 * 7 * 7)
        '''
        LeakyReLU
        '''
        self.L_ReLU = nn.LeakyReLU(0.1)
        '''
        Dropout alleviate overfitting
        '''
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, input_img):
        '''
        build up the network
        execpt final layer, activation functions are all Leaky ReLU
        final layer, activation function is linear activation function, i.e no activation function
        '''
        L11 = self.L_ReLU(self.conv1(input_img))
        L12 = self.mp1(L11)

        L21 = self.L_ReLU(self.conv2(L12))
        L22 = self.mp2(L21)

        L31 = self.L_ReLU(self.conv3(L22))
        L32 = self.L_ReLU(self.conv4(L31))
        L33 = self.L_ReLU(self.conv5(L32))
        L34 = self.L_ReLU(self.conv6(L33))
        L35 = self.mp3(L34)

        L41 = self.L_ReLU(self.conv7(L35))
        L42 = self.L_ReLU(self.conv8(L41))
        L43 = self.L_ReLU(self.conv9(L42))
        L44 = self.L_ReLU(self.conv10(L43))
        L45 = self.L_ReLU(self.conv11(L44))
        L46 = self.L_ReLU(self.conv12(L45))
        L47 = self.L_ReLU(self.conv13(L46))
        L48 = self.L_ReLU(self.conv14(L47))
        L49 = self.L_ReLU(self.conv15(L48))
        L410 = self.L_ReLU(self.conv16(L49))
        L411 = self.mp4(L410)

        L51 = self.L_ReLU(self.conv17(L411))
        L52 = self.L_ReLU(self.conv18(L51))
        L53 = self.L_ReLU(self.conv19(L52))
        L54 = self.L_ReLU(self.conv20(L53))
        L55 = self.L_ReLU(self.conv21(L54))
        L56 = self.L_ReLU(self.conv22(L55))

        L61 = self.L_ReLU(self.conv23(L56))
        L62 = self.L_ReLU(self.conv24(L61))

        L63 = self.dropout(self.L_ReLU(self.fc1(L62.view(L62.size(0), -1))))  # L62.size(0) = b_s

        L64 = self.fc2(L63)
        final_output = L64.view(-1, 7, 7, 30)
        return final_output


def __init__weights(network):
    for layer in network.modules():
        if isinstance(layer, nn.Conv2d):
            n = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
            layer.weight.data.normal_(0, math.sqrt(2.0 / n))  # normalize(mean=0, std=...)
            if layer.bias is not None:
                layer.bias.data.zero_()
        elif isinstance(layer, nn.Linear):
            layer.weight.data.normal_(0, 0.001)
            layer.bias.data.zero_()


# build the net
YOLOnet = Net()
YOLOnet.apply(__init__weights)

testNet = False
if __name__ == '__main__' and testNet:
    net = Net()
    print(net)
    input = torch.randn(1, 3, 448, 448)
    net.apply(__init__weights)
    output = net(input)
    print(output)
    print(output.size())
