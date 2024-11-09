import torch
import torch.nn as nn
from lib.utils.token_utils import patch2token,token2patch

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
# class BasicConv(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
#         super(BasicConv, self).__init__()
#         self.out_channels = out_planes
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
#
#     def forward(self, x):
#         x = self.conv(x)
#         return x

# class MultiScaleDWConv(nn.Module):
#     def __init__(self, dim, scale=(1, 3, 5, 7)):
#         super().__init__()
#         self.scale = scale
#         self.channels = []
#         self.proj = nn.ModuleList()
#         for i in range(len(scale)):
#             if i == 0:
#                 channels = dim - dim // len(scale) * (len(scale) - 1)
#             else:
#                 channels = dim // len(scale)
#             conv = nn.Conv2d(channels, channels,
#                              kernel_size=scale[i],
#                              padding=scale[i] // 2,
#                              groups=channels)
#             self.channels.append(channels)
#             self.proj.append(conv)
#
#     def forward(self, x):
#         x = torch.split(x, split_size_or_sections=self.channels, dim=1)
#         out = []
#         for i, feat in enumerate(x):
#             out.append(self.proj[i](feat))
#         x = torch.cat(out, dim=1)
#         return x
class AFF(nn.Module):
    '''
        输入: B C H W
        输出：B C H W
    '''

    def __init__(self, in_features,
                 hidden_features=None,
                 out_features=None,
                 r=4):
        super(AFF, self).__init__()
        inter_channels = int(in_features // r)
        # hidden_features = hidden_features or in_features
        # out_features = out_features or in_features
        # 1
        # self.linear = nn.Linear(dim * 2, self.channels)
        # 2
        # self.dwconv = MultiScaleDWConv(hidden_features)

        # kernel_size = 7
        self.channels = in_features
        # self.conv = BasicConv(channels, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2)  # Adjusted channels here
        self.local_att = nn.Sequential(
            nn.Conv2d(self.channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, self.channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, self.channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = x + wei
        return xo
# x = torch.ones([2,768,16,16])
# m = AFF(768,768*2)
# o = m(x)
# print(o.shape)

        # z_v = x_v[:, lens_z:, :]
        # x_v = x_v[:, :lens_z, :]
        # z_i = x_i[:, lens_z:, :]
        # x_i = x_i[:, :lens_z, :]
        #
        # z_v = token2patch(z_v)
        # x_v = token2patch(x_v)
        # z_i = token2patch(z_i)
        # x_i = token2patch(x_i)
        #
        # # 融合search
        # x = torch.cat((x_v,x_i),dim=1)  # ([2, 1536, 16, 16])
        # x = x.flatten(2).transpose(1, 2).contiguous()  #  ([2, 256, 1536])
        # xa = self.linear(x)  # ([2, 256, 768])
        # xa = token2patch(xa)
        # # xa = x_v + x_i
        # xa_l = self.local_att(xa)
        # xa_g = self.global_att(xa)
        # xa_lg = xa_g + xa_l
        # xa_wei = self.sigmoid(xa_lg)
        # x_v_sum = xa_wei + x_v
        # x_i_sum = xa_wei + x_i
        # xo = torch.cat((x_v_sum,x_i_sum),dim=1)
        #
        #
        # # 融合template
        # # z = torch.cat((z_v, z_i), dim=1)  # ([2, 1536, 16, 16])
        # # z = z.flatten(2).transpose(1, 2).contiguous()  # ([2, 256, 1536])
        # # za = self.linear(z)  # ([2, 256, 768])
        # # za = token2patch(za)
        # za = z_v + z_i
        # za_l = self.local_att1(za)
        # za_g = self.global_att1(za)
        # za_lg = za_g + za_l
        # za_wei = self.sigmoid(za_lg)
        # z_o = 2 * z_v * za_wei + 2 * z_i * (1 - za_wei)
        #
        # # 合并融合后的特征
        # x_z = patch2token(z_o)
        # x_o = patch2token(x_o)
        # x_o = torch.cat((x_z, x_o), dim=1)
        #
        # return x_o

class adapt_fusion(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.channels = dim
        self.conv = nn.Conv2d(self.channels*2,self.channels,1)
        self.lff = AFF(dim)
        self.mlp = Mlp(in_features=self.channels * 2,hidden_features=self.channels * 2,
                       out_features=self.channels,act_layer=nn.GELU)
    def forward(self,x_v):
        lens_z = 64
        z_v = x_v[:, :lens_z, :]
        x_v = x_v[:, lens_z:, :]

        z_v = token2patch(z_v)
        x_v = token2patch(x_v)

        x = x_v.flatten(2).transpose(1, 2).contiguous()  #  ([2, 256, 1536])
        xa = self.lff(x)
        x = xa + x_v
        res = self.conv(x).flatten(2).transpose(1, 2).contiguous()  # ([2, 768, 16, 16])
        x = x.flatten(2).transpose(1, 2).contiguous()  # ([2, 256, 1536])
        x = self.mlp(x) + res
        x = x.transpose(1, 2)

        # z = torch.cat((z_v, z_i), dim=1)  # ([2, 1536, 16, 16])
        # z = z.flatten(2).transpose(1, 2).contiguous()  # ([2, 256, 1536])
        # za = self.lff(z)
        # z_v_sum = za + z_v
        # z_i_sum = za + z_i
        # z = torch.cat((z_v_sum, z_i_sum), dim=1)
        # res_z = self.conv(z).flatten(2).transpose(1, 2).contiguous()  # ([2, 768, 16, 16])
        # z = z.flatten(2).transpose(1, 2).contiguous()  # ([2, 256, 1536])
        # z = self.mlp(z) + res_z
        # z = z.transpose(1, 2)
        xo = torch.cat((z_v,x),dim=1)
        return xo

x = torch.ones([2,320,768])
m = adapt_fusion(768)
o = m(x)
print(o.shape)