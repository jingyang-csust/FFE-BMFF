import torch
from torch import nn
from test_module.epsanet import EPSABlock
from test_module.lsa import LSA
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
        self.block1_x = EPSABlock(self.channels,self.channels)
        self.block2_x = LSA(self.channels,self.channels)
        self.block1_z = EPSABlock(self.channels,self.channels)
        self.block2_z = LSA(self.channels,self.channels)
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

        x = x_v + x_i
        x1 = self.block1_x(x)
        x2 = self.block2_x(x,16,16)
        x2 = token2patch(x2)
        xo_tmp = x1 + x2
        xo1 = xo_tmp + x_v
        xo2 = xo_tmp + x_i
        xo = xo1 + xo2
        xo = patch2token(xo)
        xo = xo + self.norm(self.mlp1(xo))

        z = z_v + z_i
        z1 = self.block1_z(z)
        z2 = self.block2_z(z,8,8)
        z2 = token2patch(z2)
        zo_tmp = z1 + z2
        zo1 = zo_tmp + z_v
        zo2 = zo_tmp + z_i
        zo = zo1 + zo2
        zo = patch2token(zo)
        zo = zo + self.norm(self.mlp1(zo))

        x = torch.cat((zo,xo),dim=1)
        return x

x = torch.ones(2,320,768)
m = fuision(768)
o = m(x,x,64)
print(o.shape)