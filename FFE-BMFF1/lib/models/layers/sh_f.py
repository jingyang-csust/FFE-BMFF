import torch
from torch import nn

from lib.models.layers.las import LSA
from lib.models.layers.shaf import ShuffleAttention
from lib.utils.token_utils import token2patch, patch2token

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
class fuision(nn.Module):
    """
    输入；B N C
    输出：B N C
    """
    def __init__(self,dim):
        super().__init__()
        self.channels = dim
        # self.linear = nn.Linear(self.channels * 2, self.channels)

        self.block1_x = LSA(self.channels,self.channels)
        self.block2_x = ShuffleAttention(self.channels)
        self.block1_z = LSA(self.channels,self.channels)
        self.block2_z = ShuffleAttention(self.channels)
        self.mlp1 = Mlp(in_features=self.channels, hidden_features=self.channels * 2, act_layer=nn.GELU)
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

        # x = torch.cat((x_v, x_i), dim=1)  # ([2, 1536, 16, 16])
        # x = x.flatten(2).transpose(1, 2).contiguous()  # ([2, 256, 1536])
        # # print(x.shape) ([2, 256, 1536])
        # x = self.linear(x)  # ([2, 256, 768])
        # x = token2patch(x)  # ([2, 768, 16， 16])
        x = x_v + x_i
        B, C, H, W = x.shape
        x1 = self.block1_x(x,H,W)
        x2 = self.block2_x(x)
        xo_tmp = x1 + x2
        xo1 = xo_tmp + x_v
        xo2 = xo_tmp + x_i
        xo = xo1 + xo2
        xo = patch2token(xo)
        xo = xo + self.norm(self.mlp1(xo))

        z = z_v + z_i
        B, C, H_z, W_z = z.shape
        z1 = self.block1_z(z,H_z,W_z)
        z2 = self.block2_z(z)
        zo_tmp = z1 + z2
        zo1 = zo_tmp + z_v
        zo2 = zo_tmp + z_i
        zo = zo1 + zo2
        zo = patch2token(zo)
        zo = zo + self.norm(self.mlp1(zo))

        x = torch.cat((zo,xo),dim=1)
        return x

# class MS_Fusion(nn.Module):
#     def __init__(self, dim=768 // 2, xavier_init=False):
#         super().__init__()
#
#         self.adapter_down = nn.Linear(768, dim)
#         self.adapter_mid = fuision(dim)
#         self.adapter_up = nn.Linear(dim, 768)
#
#         #nn.init.xavier_uniform_(self.adapter_down.weight)
#         # nn.init.zeros_(self.adapter_mid.bias)
#         # nn.init.zeros_(self.adapter_mid.weight)
#
#         # nn.init.zeros_(self.adapter_mid_upscale.bias)
#         # nn.init.zeros_(self.adapter_mid_upscale.weight)
#         nn.init.zeros_(self.adapter_up.weight)
#         nn.init.zeros_(self.adapter_up.bias)
#         nn.init.zeros_(self.adapter_down.weight)
#         nn.init.zeros_(self.adapter_down.bias)
#
#         #self.act = QuickGELU()
#         self.dropout = nn.Dropout(0.1)
#         self.dim = dim
#
#     def forward(self, x,xi,lens_x):
#         # B, N, C = x.shape
#         x_down = self.adapter_down(x)
#         xi_down = self.adapter_down(xi)
#         x_down = self.adapter_mid(x_down,xi_down,lens_x)
#         x_down = self.dropout(x_down)
#         x_up = self.adapter_up(x_down)
#
#         return x_up
# x = torch.ones(2,320,768)
# m = MS_Fusion(768)
# o = m(x,x,64)
# print(o.shape)