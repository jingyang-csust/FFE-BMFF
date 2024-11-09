import torch
# from timm.layers import Mlp
from torch import nn
import math

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()  # GELU 激活函数
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class SGBlock(nn.Module):
    def __init__(self, dim, mlp_ratio, h=20, w=20, SC=768,drop=0.,norm_layer=nn.LayerNorm):
        super().__init__()
        mlp_hidden_dim = int(mlp_ratio*dim)
        self.w = w
        self.h = h
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        # self.complex_weight = nn.Parameter(torch.randn(dim, h, (w//2)+1, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(h, w // 2 + 1, dim, 2, dtype=torch.float32) * 0.02)
        self.basic_conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=1),
            nn.BatchNorm2d(dim)
        )
        # print(dim)
    def forward(self, x,lens_x,spatial_size=None):
        # print(f"lens_x : {lens_x}") 256
        x = self.norm1(x)

        z = x[:, 0:lens_x, :]
        z = z.permute(0, 2, 1).unsqueeze(3)
        z = self.basic_conv1(z)
        z = z.squeeze(3).permute(0, 2, 1)

        x = x[:, lens_x:, :]
        # print(f"z.shape{z.shape},x.shape{x.shape}")
        B, n, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(n))
        else:
            a, b = spatial_size
        x = x.view(B, a, b, C)
        x = x.to(torch.float32)
        # print(f"x.shape:{x.shape}")
        x_fft = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        # print(f"weight:{weight.shape} x_fft:{x_fft.shape}")
        x_fft = x_fft * weight
        x = torch.fft.irfft2(x_fft, s=(a, b), dim=(1, 2), norm='ortho')
        x = x.reshape(B, n, C)
        x = torch.cat([z, x], dim=1)
        x = self.norm2(x)
        x = self.mlp(x)
        return x

class Block(nn.Module):
    def __init__(self, dim,mlp_ratio, h, w, drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm1 = norm_layer(dim)
        self.stb_x = SGBlock(dim, h=h, w=w,mlp_ratio=mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)


    def forward(self,x,lens_x):
        res_x = x
        x = self.stb_x(x,lens_x)
        x = res_x + x
        return x

# fft = SGBlock(dim=768,mlp_ratio=4,h=16,w=16, norm_layer=nn.LayerNorm)
# print(fft)
# x = torch.ones([2, 320, 768])
# out = fft(x,64)
# print(out.shape)