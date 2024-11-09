import torch
from torch import nn

from lib.utils.token_utils import token2patch, patch2token


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, lens_z):
        x = x[:,lens_z:,:]
        z = x[:,:lens_z,:]
        x = token2patch(x)
        B, C, H, W = x.size()
        x = patch2token(x)
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        z = token2patch(z)
        B, C, h, w = z.size()
        z = patch2token(z)
        B, N, C = z.shape
        z = z.transpose(1, 2).view(B, C, h, w).contiguous()
        z = self.dwconv(z)
        z = z.flatten(2).transpose(1, 2)

        x = torch.cat((z,x),dim=1)

        return x

class ConvolutionalGLU1(nn.Module):
    """
    Input:B C H W
    Output:B C H W
    """
    def __init__(self, in_features=768, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Linear(in_features, hidden_features * 2)  # 512
        self.norm = nn.LayerNorm(in_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        lens_z = 64
        x_tmp = self.norm(x)
        x_tmp, v = self.fc1(x_tmp).chunk(2, dim=-1)  # ([2, 256, 512])
        x_tmp = self.act(self.dwconv(x_tmp,lens_z)) * v
        x_tmp = self.drop(x_tmp)
        x_tmp = self.fc2(x_tmp)
        x_tmp = self.drop(x_tmp)
        x = x + x_tmp
        return x  # ([2, 256, 768])

class MS_MLP(nn.Module):
    def __init__(self, dim=8, xavier_init=False):
        super().__init__()

        self.adapter_down = nn.Linear(768, dim)
        self.adapter_mid = ConvolutionalGLU1(dim)
        self.adapter_up = nn.Linear(dim, 768)


        #self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        # B, N, C = x.shape
        x_down = self.adapter_down(x)
        x_down = self.adapter_mid(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)

        return x_up
#
# x = torch.ones(2,320,768)
# m = ConvolutionalGLU1(768)
# o = m(x,64)
# print(o.shape)
#
#
# class se_block(nn.Module):
#     def __init__(self, channel, ratio=16):
#         super(se_block, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均池化
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // ratio, bias=False),  # 全连接
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // ratio, channel, bias=False),  # 全连接
#             nn.Sigmoid()
#         )
#
#     def forward(self, x,lens_z):
#         x = x[:,lens_z:,:]
#         z = x[:,:lens_z,:]
#
#         x = token2patch(x)
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         x = patch2token(x * y)
#
#         z = token2patch(z)
#         b, c, _, _ = z.size()
#         zy = self.avg_pool(z).view(b, c)
#         zy = self.fc(zy).view(b, c, 1, 1)
#         z = patch2token(z * zy)
#
#         return torch.cat((z,x),dim=1)
#
# class ConvolutionalGLU2(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         hidden_features = int(2 * hidden_features / 3)
#         self.norm = nn.LayerNorm(in_features)
#         self.fc1 = nn.Linear(in_features, hidden_features * 2)  # 512
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features * 2, out_features)
#         self.drop = nn.Dropout(drop)
#         self.se = se_block(out_features)
#
#     def forward(self, x,lens_z):
#         x = self.norm(x)
#         x1= self.fc1(x)
#         x1 = self.act(x1)
#         x1 = self.drop(x1)  # ([2, 256, 1024])
#         x1 = self.fc2(x1)
#         x2 = self.se(x1,lens_z)
#         x = x1 * x2
#         x = self.drop(x)
#         return x
# x = torch.ones(2,768,16,16)
# m = ConvolutionalGLU2(768)
# o = m(x)
# print(o.shape)

