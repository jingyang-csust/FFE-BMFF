import torch
from einops import rearrange
from torch import nn
import math
from wave_p import DWT_2D, IDWT_2D  # Assuming wave_p.py contains the wavelet transform code

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class SGBlock(nn.Module):
    def __init__(self, dim, mlp_ratio, h=20, w=20, SC=768, drop=0., norm_layer=nn.LayerNorm, wavelet='haar'):
        super().__init__()
        mlp_hidden_dim = int(mlp_ratio * dim)
        self.w = w
        self.h = h
        self.norm1 = norm_layer(dim)
        self.dwt = DWT_2D(wavelet)
        self.i_dwt = IDWT_2D(wavelet)
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.basic_conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=1),
            nn.BatchNorm2d(dim)
        )
        self.basic_conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x, lens_x,H,W, spatial_size=None):
        res_z_x = self.basic_conv2(x)

        z = x[:, 0:lens_x, :]
        x = x[:, lens_x:, :]
        x = self.norm1(x)
        res = x
        res = rearrange(res, 'b (h w) c -> b c h w ', h=H, w=W)
        res = self.basic_conv(res)
        res = rearrange(res, 'b c h w -> b (h w) c')
        # print(f"res.shape{res.shape}")
        B, n, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(n))
        else:
            a, b = spatial_size
        x = x.view(B, a, b, C).permute(0, 3, 1, 2)
        x = x.to(torch.float32)
        x_dwt = self.dwt(x)
        print(x_dwt.shape)
        x_i_dwt = self.i_dwt(x_dwt)
        print(x_i_dwt.shape)
        x_i_dwt = x_i_dwt.view(B, -1, x_i_dwt.size(-2) * x_i_dwt.size(-1)).transpose(1, 2)
        print(f"x_i_dwt{x_i_dwt.shape}")

        x = self.norm2(x_i_dwt + res)
        x = torch.cat([z, x], dim=1)
        x = self.mlp(x)
        return x

fft = SGBlock(dim=768, mlp_ratio=4, h=16, w=16, norm_layer=nn.LayerNorm)
x = torch.ones([2, 320, 768])
out = fft(x, lens_x=64,H=16,W=16)
print(out.shape)



# class Block(nn.Module):
#     def __init__(self, dim,mlp_ratio, h, w, drop=0., norm_layer=nn.LayerNorm):
#         super().__init__()
#         mlp_hidden_dim = int(dim * mlp_ratio)
#
#         self.norm1 = norm_layer(dim)
#         self.stb_x = SGBlock(dim, h=h, w=w,mlp_ratio=mlp_ratio)
#         self.norm2 = norm_layer(dim)
#         self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
#
#
#     def forward(self,x,lens_x):
#         res_x = x
#         x = self.norm2(self.stb_x(self.norm1(x),lens_x))
#         x = res_x + self.mlp(x)
#         return x