import torch
from torch import nn
import math
from lib.utils.token_utils import token2patch,patch2token

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
        self.act = nn.ReLU()


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
        x_fft_real = self.act(self.norm1(x_fft_real))
        x_fft_imag = self.act(self.norm2(x_fft_imag))
        x_fft = torch.complex(x_fft_real, x_fft_imag).permute(0, 2, 3, 1)

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
        self.basic1 = basic_conv(hidden_dim, hidden_dim)
        self.basic2 = basic_conv(hidden_dim, hidden_dim)
        self.stb_x = SpectralGatingNetwork(hidden_dim, h=16, w=16)
        self.stb_z = SpectralGatingNetwork(hidden_dim, h=8, w=8)
        self.conv1x1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.conv1x1_z = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0)
        # self.conv3x3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm2d(hidden_dim)
        self.norm_z = nn.BatchNorm2d(hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.adapter_up = nn.Linear(hidden_dim, dim)

        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

    def forward(self, x):
        x = self.adapter_down(x)
        z = x[:,:64,:]
        x = x[:,64:,:]

        z = token2patch(z)
        x = token2patch(x)

        x1 = self.basic1(x)
        res_x1 = x1  # ([2, 8, 16, 16])
        x1 = self.stb_x(x1)  # ([2, 16, 16, 8])
        xo = x1 + res_x1
        xo = self.conv1x1(xo)
        xo = patch2token(xo)
        # ------------------------ #
        # x2 = x
        # x2 = self.conv3x3(x2)
        # xo = xo + x2
        # xo = self.norm(xo)
        # xo = patch2token((xo))
        # xo = self.act(xo)
        # ------------------------ #
        z1 = self.basic2(z)
        res_z1 = z1  # ([2, 8, 16, 16])
        z1 = self.stb_z(z1)  # ([2, 16, 16, 8])
        zo = z1 + res_z1
        zo = self.conv1x1_z(zo)
        zo = patch2token(zo)
        xo_f = torch.cat((zo,xo),dim=1)

        xo_f = self.adapter_up(xo_f)
        return xo_f

# x = torch.ones(2,320,768)
# m = SGBlock(768)
# o = m(x)
# print(o.shape)