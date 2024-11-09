from timm.layers import Mlp
from torch import nn

from lib.models.layers.FFT import SpectralGatingNetwork


class SpectBlock(nn.Module):
    def __init__(self, dim,mlp_ratio, h, w,ff_dropout=0., drop=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        # self.layers = nn.ModuleList([])
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.stb_x = SpectralGatingNetwork(dim, h=h, w=w)
        self.stb_xi = SpectralGatingNetwork(dim, h=h, w=w)

    def forward(self, x, xi,lens_x):
        res_x = x
        x = self.norm2(self.stb_x(self.norm1(x)))
        x = res_x + self.mlp0(x)

        res_xi = xi
        xi = self.norm2(self.stb_xi(self.norm1(xi)))
        xi = res_xi + self.mlp0(xi)

        return x,xi