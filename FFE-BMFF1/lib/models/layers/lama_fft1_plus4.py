import torch
from timm.layers import DropPath
from torch import nn
import math

from lib.models.layers.lama1_plus import Mlp
from lib.models.layers.lama1_plus3 import Attention_Module
from lib.utils.token_utils import token2patch,patch2token

class FeedForward(nn.Module):
    """MLP"""

    def __init__(self, dim, hidden_dim, dropout=0., drop_path=0.):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.drop_path(self.net(x))

class SpectralGatingNetwork(nn.Module):
    def __init__(self, dim, h=20, w=20, SC=768, drop=0.):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        # Initialize complex weights
        self.complex_weight = nn.Parameter(torch.randn(h, w // 2 + 1, dim, 2, dtype=torch.float32) * 0.02)
        # Initialize 1x1 convolution and GELU activation
        # self.conv1x1_real = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        # self.conv1x1_imag = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        # self.act = nn.ReLU()


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
        # x_fft_real = x_fft.real.permute(0, 3, 1, 2)
        # x_fft_imag = x_fft.imag.permute(0, 3, 1, 2)

        # Apply 1x1 convolution to real and imaginary parts separately
        # x_fft_real = self.conv1x1_real(x_fft_real)
        # x_fft_imag = self.conv1x1_imag(x_fft_imag)
        # x_fft_real = self.act(self.norm1(x_fft_real))
        # x_fft_imag = self.act(self.norm2(x_fft_imag))
        # x_fft = torch.complex(x_fft_real, x_fft_imag).permute(0, 2, 3, 1)

        # Apply spectral gating
        weight = torch.view_as_complex(self.complex_weight)
        x_fft = x_fft * weight
        x = torch.fft.irfft2(x_fft, s=(a, b), dim=(1, 2), norm='ortho')

        return x.permute(0,3,2,1)


class SGBlock(nn.Module):
    def __init__(self, dim = 768, drop=0.):
        super().__init__()
        hidden_dim = 8
        self.adapter_down = nn.Linear(dim, hidden_dim)
        self.stb_x = SpectralGatingNetwork(hidden_dim, 16, 16)
        self.stb_z = SpectralGatingNetwork(hidden_dim, 8, 8)
        self.norm_x1 = nn.LayerNorm(hidden_dim)
        self.norm_x2 = nn.LayerNorm(hidden_dim)
        self.mlp_x = FeedForward(hidden_dim,hidden_dim * 2)
        self.norm_z1 = nn.LayerNorm(hidden_dim)
        self.norm_z2 = nn.LayerNorm(hidden_dim)
        self.mlp_z = FeedForward(hidden_dim,hidden_dim * 2)
        # self.sum = Attention_Module(hidden_dim)
        # self.mlp2 = Mlp(hidden_dim)
        # self.norm2 = nn.LayerNorm(hidden_dim)
        self.adapter_up = nn.Linear(hidden_dim, dim)

        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

    def forward(self, x):
        x = self.adapter_down(x)
        z = x[:,:64,:]
        x = x[:,64:,:]

        raw_x = x
        raw_z = z
        z = token2patch(z)
        x = token2patch(x)

        x1 = self.norm_x1(patch2token(x))
        x1 = self.stb_x(token2patch(x1))  # ([2, 16, 16, 8])
        x1 = self.norm_x2(patch2token(x1))
        x1 = self.mlp_x(x1)
        xo = x1 + raw_x

        z1 = self.norm_x1(patch2token(z))
        z1 = self.stb_z(token2patch(z1))  # ([2, 16, 16, 8])
        z1 = self.norm_x2(patch2token(z1))
        z1 = self.mlp_x(z1)
        zo = z1 + raw_z

        xo_f = torch.cat((zo,xo),dim=1)
        xo_f = self.adapter_up(xo_f)
        return xo_f


# x = torch.ones(2,320,768)
# m = SGBlock(768)
# o = m(x)
# print(o.shape)