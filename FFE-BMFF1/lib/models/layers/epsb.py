import torch
import torch.nn as nn
import math

from lib.utils.token_utils import token2patch, patch2token
from test_module.SE_weight_module import  SEWeightModule

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                   stride=1, padding=0, groups=groups)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
class PSAModule(nn.Module):
    """
    输入：B C H W
    输出：B C H W
    """
    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(PSAModule, self).__init__()
        self.conv_1 = DepthwiseSeparableConv(inplans, planes // 4, kernel_size=conv_kernels[0],
                                             padding=conv_kernels[0] // 2, stride=stride, groups=conv_groups[0])
        self.conv_2 = DepthwiseSeparableConv(inplans, planes // 4, kernel_size=conv_kernels[1],
                                             padding=conv_kernels[1] // 2, stride=stride, groups=conv_groups[1])
        self.conv_3 = DepthwiseSeparableConv(inplans, planes // 4, kernel_size=conv_kernels[2],
                                             padding=conv_kernels[2] // 2, stride=stride, groups=conv_groups[2])
        self.conv_4 = DepthwiseSeparableConv(inplans, planes // 4, kernel_size=conv_kernels[3],
                                             padding=conv_kernels[3] // 2, stride=stride, groups=conv_groups[3])
        self.se = SEWeightModule(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        # print(feats.shape) # ([2, 768, 16, 16])
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])
        # print(feats.shape) # ([2, 4, 192, 16, 16])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)
        # print(x4_se.shape) # ([2, 192, 1, 1])

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        # print(x_se.shape) # ([2, 768, 1, 1])

        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        # print(attention_vectors.shape) # ([2, 4, 192, 1, 1])
        attention_vectors = self.softmax(attention_vectors)
        # print(attention_vectors.shape) 3 ([2, 4, 192, 1, 1])
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out

# x = torch.ones([2,768,16,16])
# m = PSAModule(768,768)
# o = m(x)
# print(o.shape)


class EPSABlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, conv_kernels=[3, 5, 7, 9],
                 conv_groups=[1, 4, 8, 16]):
        super(EPSABlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = PSAModule(planes, planes, stride=stride, conv_kernels=conv_kernels, conv_groups=conv_groups)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.t_downsample = nn.Conv2d(in_channels=planes * self.expansion, out_channels=inplanes, kernel_size=1, stride=1, padding=0)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # print(out.shape) # ([2, 3072, 16, 16])
        # print(identity.shape) # ([2, 768, 16, 16])
        out = self.t_downsample(out)
        out += identity
        out = self.relu(out)
        return out