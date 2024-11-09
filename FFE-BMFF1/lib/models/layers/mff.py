import torch
import torch.nn as nn
from lib.utils.token_utils import patch2token,token2patch


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        return x

class MultiScaleDWConv(nn.Module):
    def __init__(self, dim, scale=(1, 3, 5, 7)):
        super().__init__()
        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)
            conv = nn.Conv2d(channels, channels,
                             kernel_size=scale[i],
                             padding=scale[i] // 2,
                             groups=channels)
            self.channels.append(channels)
            self.proj.append(conv)

    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x
class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, in_features,
                 hidden_features=None,
                 out_features=None,
                 r=2):
        super(AFF, self).__init__()
        inter_channels = int(in_features // r)
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.dwconv = MultiScaleDWConv(inter_channels)
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, inter_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.norm0 = nn.BatchNorm2d(inter_channels)
        self.act = nn.BatchNorm2d(inter_channels)
        self.fc2 = nn.Sequential(
            nn.Conv2d(inter_channels, in_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_features),
        )
        # kernel_size = 7
        # self.conv = BasicConv(channels, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2)  # Adjusted channels here

        self.local_att = nn.Sequential(
            nn.Conv2d(in_features, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, in_features, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_features),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_features, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, in_features, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_features),
        )

        self.local_att1 = nn.Sequential(
            nn.Conv2d(in_features, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, in_features, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_features),
        )

        self.global_att1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_features, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, in_features, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_features),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x,lens_z):
        x_z = x[:,:lens_z,:]
        x_x = x[:,lens_z:,:]
        x_z = token2patch(x_z)
        x_x = token2patch(x_x)

        # res_x = x_x
        # x_x = self.dwconv(x_x)
        # 融合search
        res_x = x_x
        x_x = self.fc1(x_x) # 2 2 16 16
        x_x = self.dwconv(x_x) + x_x
        x_x = self.norm0(self.act(x_x))
        x_x = self.fc2(x_x)

        xa = x_x + res_x
        xa_l = self.local_att(xa)
        xa_g = self.global_att(xa)
        xa_lg = xa_g + xa_l
        xa_wei = self.sigmoid(xa_lg)
        x_o = 2 * x_x * xa_wei + 2 * res_x * (1 - xa_wei)

        # res_z = x_z
        # x_z = self.dwconv(x_z)
        # 融合template
        res_z = x_z
        x_z = self.fc1(x_z)
        x_z = self.dwconv(x_z) + x_z
        x_z = self.norm0(self.act(x_z))
        x_z = self.fc2(x_z)

        za = x_z + res_z
        za_l = self.local_att1(za)
        za_g = self.global_att1(za)
        za_lg = za_g + za_l
        za_wei = self.sigmoid(za_lg)
        z_o = 2 * x_z * za_wei + 2 * res_z * (1 - za_wei)

        # 合并融合后的特征
        x_z = patch2token(z_o)
        x_o = patch2token(x_o)
        x_o = torch.cat((x_z,x_o),dim=1)

        return x_o

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)



class MFF_adapter(nn.Module):
    def __init__(self, dim=384, xavier_init=False):
        super().__init__()

        self.dropout = nn.Dropout(0.1)

        self.adapter_down = nn.Linear(768, dim)
        self.adapter_up = nn.Linear(dim, 768)
        self.adapter_mid = AFF(in_features=dim)

        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        #self.act = QuickGELU()
        self.dim = dim

    def forward(self, x,lens_z):
        B, N, C = x.shape
        x_down = self.adapter_down(x)
        #x_down = self.act(x_down)
        x_down = self.adapter_mid(x_down,lens_z)
        #x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)
        #print("return adap x", x_up.size())
        return x_up
#
# x  = torch.ones([2,320,768])
# channels = x.shape[2]
# model=MFF_adapter(channels)
# output = model(x,64)
# print(output.shape)