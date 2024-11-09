import torch
from torch import nn
from lib.models.layers.epsb import PSAModule
from lib.models.layers.fudion9 import Attention
from lib.models.layers.las import LSA
from lib.models.layers.shaf import ShuffleAttention
from lib.utils.token_utils import token2patch, patch2token
from test_module.dwconv import MultiScaleDWConv


class Mlp(nn.Module):  ### MS-FFN
    """
    Input: Tensor with shape [B, C, H, W].
    Output: Tensor with shape [B, C, H, W].
    """

    def __init__(self,
                 in_features,
                 h,
                 w,
                 hidden_features=None,
                 out_features=None,
                 drop=0, ):
        super().__init__()
        self.h = h
        self.w = w
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(hidden_features),
        )
        self.dwconv = MultiScaleDWConv(hidden_features,self.h,self.w)
        self.act = nn.GELU()
        self.norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, in_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_features),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)

        x = self.dwconv(x) + x
        x = self.norm(self.act(x))

        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x
class fuision(nn.Module):
    """
    输入；B N C
    输出：B N C
    """
    def __init__(self,dim):
        super().__init__()
        self.channels = dim
        self.block1_x = Attention(self.channels)
        self.linear_x = nn.Linear(dim * 2, self.channels)
        self.block2_x = LSA(self.channels,self.channels)
        self.block1_z = Attention(self.channels)
        self.linear_z = nn.Linear(dim * 2, self.channels)

        self.block2_z = LSA(self.channels,self.channels)
        self.mlp1 = Mlp(in_features=self.channels, hidden_features=self.channels * 2,out_features=self.channels,h=16,w=16)
        self.mlp2 = Mlp(in_features=self.channels, hidden_features=self.channels * 2,out_features=self.channels,h=8,w=8)
        self.norm = nn.LayerNorm(self.channels)

    def forward(self,x_v,x_i,lens_z):
        z_v = x_v[:, :lens_z, :]
        x_v = x_v[:, lens_z:, :]
        z_i = x_i[:, :lens_z, :]
        x_i = x_i[:, lens_z:, :]

        z_v = token2patch(z_v)
        x_v = token2patch(x_v)
        z_i = token2patch(z_i)
        x_i = token2patch(x_i)

        # x = x_v + x_i
        x = torch.cat((x_v, x_i), dim=1)  # ([2, 1536, 16, 16])
        x = x.flatten(2).transpose(1, 2).contiguous()  # ([2, 256, 1536])
        # print(x.shape) ([2, 256, 1536])
        x = self.linear_x(x)  # ([2, 256, 768])
        x = token2patch(x)  # ([2, 768, 16， 16])
        x1 = self.block1_x(x)
        x = patch2token(x)
        x2 = self.block2_x(x,16,16)
        x2 = token2patch(x2)
        xo_tmp = x1 + x2
        xo1 = xo_tmp * x_v
        xo2 = xo_tmp * x_i
        # xo = torch.cat((xo1, xo2), dim=1)  # ([2, 1536, 16, 16])
        # xo = xo.flatten(2).transpose(1, 2).contiguous()  # ([2, 256, 1536])
        xo = xo1 + xo2
        # xo = patch2token(xo)
        xo = xo + self.mlp1(token2patch(self.norm(patch2token(xo))))
        xo = patch2token(xo)

        # z = z_v + z_i
        z = torch.cat((z_v, z_i), dim=1)  # ([2, 1536, 16, 16])
        z = z.flatten(2).transpose(1, 2).contiguous()  # ([2, 256, 1536])
        # print(x.shape) ([2, 256, 1536])
        z = self.linear_z(z)  # ([2, 256, 768])
        z = token2patch(z)  # ([2, 768, 16， 16])
        z1 = self.block1_z(z)
        z = patch2token(z)
        z2 = self.block2_z(z,8,8)
        z2 = token2patch(z2)
        zo_tmp = z1 + z2
        zo1 = zo_tmp * z_v
        zo2 = zo_tmp * z_i
        # zo = torch.cat((zo1, zo2), dim=1)  # ([2, 1536, 16, 16])
        zo = zo1 + zo2
        # zo = zo.flatten(2).transpose(1, 2).contiguous()  # ([2, 256, 1536])
        zo = zo + self.mlp2(token2patch(self.norm(patch2token(zo))))
        zo = patch2token(zo)

        x = torch.cat((zo,xo),dim=1)
        return x

class MS_Fusion(nn.Module):
    def __init__(self, dim=768 // 2, xavier_init=False):
        super().__init__()

        self.adapter_down = nn.Linear(768, dim)
        self.adapter_mid = fuision(dim)
        self.adapter_up = nn.Linear(dim, 768)

        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)

        #self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x,xi,lens_x):
        # B, N, C = x.shape
        x_down = self.adapter_down(x)
        xi_down = self.adapter_down(xi)
        x_down = self.adapter_mid(x_down,xi_down,lens_x)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)

        return x_up
# x = torch.ones(2,320,768)
# m = MS_Fusion()
# o = m(x,x,64)
# print(o.shape)