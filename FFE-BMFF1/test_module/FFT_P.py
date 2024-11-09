import torch
from torch import nn
import math


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()  # GELU activation function
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class SpectralGatingNetwork(nn.Module):
    def __init__(self, dim, h=20, w=20, SC=768, drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.w = w
        self.h = h
        # Initialize complex weights
        self.complex_weight = nn.Parameter(torch.randn(h, w // 2 + 1, dim, 2, dtype=torch.float32) * 0.02)
        # Initialize 3x3 convolution and GELU activation
        self.conv3x3_real = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv3x3_imag = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.act = nn.ReLU()

    def forward(self, x, lens_x, spatial_size=None):
        z = x[:, 0:lens_x, :]
        x = x[:, lens_x:, :]
        x = self.norm1(x)
        B, n, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(n))
        else:
            a, b = spatial_size
        x = x.view(B, a, b, C)
        x = x.to(torch.float32)

        # Apply FFT
        x_fft = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        # Apply 3x3 convolution and GELU activation
        x_fft_real = x_fft.real.permute(0, 3, 1, 2)
        x_fft_imag = x_fft.imag.permute(0, 3, 1, 2)

        # Apply 3x3 convolution to real and imaginary parts separately
        x_fft_real = self.conv3x3_real(x_fft_real)
        x_fft_imag = self.conv3x3_imag(x_fft_imag)
        x_fft_real = self.act(x_fft_real)
        x_fft_imag = self.act(x_fft_imag)

        x_fft = torch.complex(x_fft_real, x_fft_imag).permute(0, 2, 3, 1)

        # Apply spectral gating
        weight = torch.view_as_complex(self.complex_weight)
        x_fft = x_fft * weight
        x = torch.fft.irfft2(x_fft, s=(a, b), dim=(1, 2), norm='ortho')

        x = x.reshape(B, n, C)
        x = torch.cat([z, x], dim=1)
        return x


class SGBlock(nn.Module):
    def __init__(self, dim, mlp_ratio, h, w, drop=0.):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.stb_x = SpectralGatingNetwork(dim, h=h, w=w)
        self.conv1x1 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.act = nn.ReLU()

    def forward(self, x, lens_x):
        x0 = self.stb_x(x,lens_x)

        B,N,C = x.shape
        x = x.permute(0, 2, 1).unsqueeze(3)
        # print(x.shape)
        x1 = self.act(self.conv3x3(x))
        x2 = self.act(self.conv1x1(x))
        x_p = x1 + x2
        x_p = x_p.permute(0, 2, 3, 1).reshape(B, N, C)
        x = self.act(x_p + x0)

        return x


# Example usage
fft = SGBlock(dim=768, mlp_ratio=4, h=16, w=16)
x = torch.ones([2, 320, 768])
x_output = fft(x, 64)
print(x.shape)
