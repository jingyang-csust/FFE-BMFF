import torch
from torch import nn
import math

from lib.models.layers.adapter_upgtaded3 import ECAAttention
from lib.utils.token_utils import token2patch,patch2token

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

class basic_conv(nn.Module):
    def __init__(self,channels,inter_channels):
        super(basic_conv,self).__init__()
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.local_att(x)
        return x

class SpectralGatingNetwork(nn.Module):
    def __init__(self, dim, h=20, w=20, SC=768, drop=0.):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        # Initialize complex weights
        self.complex_weight = nn.Parameter(torch.randn(h, w // 2 + 1, dim, 2, dtype=torch.float32) * 0.02)
        # Initialize 1x1 convolution and GELU activation
        self.conv1x1_real = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.conv1x1_imag = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()


    def forward(self, x):
        x = x.permute(0,3,2,1)
        B, a, b, C = x.shape  # ([2, 16, 16, 8])
        # if spatial_size is None:
        #     a = b = int(math.sqrt(n))
        # else:
        #     a, b = spatial_size
        # x = x.view(B, a, b, C)
        x = x.to(torch.float32)
        # print("ggg:",x.shape)
        # Apply FFT
        x_fft = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        # Apply 1x1 convolution and GELU activation
        x_fft_real = x_fft.real.permute(0, 3, 1, 2)
        x_fft_imag = x_fft.imag.permute(0, 3, 1, 2)

        # Apply 1x1 convolution to real and imaginary parts separately
        x_fft_real = self.conv1x1_real(x_fft_real)
        x_fft_imag = self.conv1x1_imag(x_fft_imag)
        x_fft_real = self.act1(self.norm1(x_fft_real))
        x_fft_imag = self.act2(self.norm2(x_fft_imag))
        x_fft = torch.complex(x_fft_real, x_fft_imag).permute(0, 2, 3, 1)

        # Apply spectral gating
        weight = torch.view_as_complex(self.complex_weight)
        x_fft = x_fft * weight
        x = torch.fft.irfft2(x_fft, s=(a, b), dim=(1, 2), norm='ortho')

        return x.permute(0,3,2,1)


class SGBlock(nn.Module):
    def __init__(self, dim = 8, drop=0.):
        super().__init__()
        self.dim = dim
        self.basic1 = basic_conv(self.dim, self.dim)
        self.basic2 = basic_conv(self.dim, self.dim)
        self.stb_x = SpectralGatingNetwork(self.dim, 16, 16)
        self.stb_z = SpectralGatingNetwork(self.dim, 8, 8)
        self.conv1x1 = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0)
        self.conv1x1_2 = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1)
        self.conv3x3_2 = nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm2d(self.dim)
        self.act = nn.ReLU(inplace=True)
        self.sum = ECAAttention(kernel_size=3)

    def forward(self, x):
        z = x[:,:64,:]
        x = x[:,64:,:]

        z = token2patch(z)
        x = token2patch(x)

        x1 = self.basic1(x)
        res_x1 = x1  # ([2, 8, 16, 16])
        x1 = self.stb_x(x1)  # ([2, 16, 16, 8])
        xo = x1 + res_x1
        xo = self.conv1x1(xo)
        xo = self.norm(xo)
        xo = patch2token((xo))
        xo = self.act(xo)
        # x2 = x
        # x2 = self.conv3x3(x2)
        # xo = token2patch(xo) + x2
        # xo = self.norm(xo)
        # xo = patch2token((xo))
        # xo = self.act(xo)


        z1 = self.basic2(z)
        res_z1 = z1  # ([2, 8, 16, 16])
        z1 = self.stb_z(z1)  # ([2, 16, 16, 8])
        zo = z1 + res_z1
        zo = self.conv1x1_2(zo)
        zo = self.norm(zo)
        zo = patch2token((zo))
        zo = self.act(zo)
        # z2 = z
        # z2 = self.conv3x3_2(z2)
        # zo = token2patch(zo) + z2
        # zo = self.norm(zo)
        # zo = patch2token((zo))
        # zo = self.act(zo)


        xo_f = torch.cat((zo,xo),dim=1)
        xo_f = self.sum(xo_f)
        return xo_f


# x = torch.ones(2,320,768)
# m = SGBlock(768)
# o = m(x)
# print(o.shape)


class SG_Block(nn.Module):
    def __init__(self, dim=768, norm_layer=nn.LayerNorm):
        super().__init__()
        self.down = nn.Linear(dim,8)
        self.up = nn.Linear(8,dim)
        self.stb_x1 = SGBlock(8)
        self.stb_x2 = SGBlock(8)
        # self.sum = ECAAttention(kernel_size=3)

        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self,x):
        x1 = self.down(x)
        res = x1
        x1 = self.stb_x1(x1)
        x_sum = x1 + res
        # x_sum = self.sum(x_sum)
        x_sum = self.up(x_sum)


        return x_sum
# x = torch.ones(2,320,768)
# m = SG_Block(768)
# o = m(x)
# print(o.shape)