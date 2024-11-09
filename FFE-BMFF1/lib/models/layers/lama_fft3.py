import torch
from timm.layers import DropPath
# from timm.layers import Mlp
from torch import nn
import math

from lib.utils.token_utils import token2patch, patch2token


class MLP(nn.Module):
    """MLP"""
    def __init__(self, in_features, hidden_features, drop=0., drop_path=0.):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_features, in_features),
            nn.Dropout(drop)
        )

    def forward(self, x):
        return self.drop_path(self.net(x))

class SGBlock(nn.Module):
    def __init__(self, dim = 8, mlp_ratio = 4,drop=0.,norm_layer=nn.LayerNorm):
        super().__init__()
        mlp_hidden_dim = int(mlp_ratio*dim)
        self.dim = dim
        self.norm1 = norm_layer(self.dim)
        self.norm2 = norm_layer(self.dim)
        self.norm3 = norm_layer(self.dim)
        self.norm4 = norm_layer(self.dim)
        self.conv3x3 = nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1)
        self.conv3x3_2 = nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1)
        self.mlp = MLP(in_features=self.dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.mlp2 = MLP(in_features=self.dim, hidden_features=mlp_hidden_dim, drop=drop)
        # self.complex_weight = nn.Parameter(torch.randn(dim, h, (w//2)+1, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(16, 16 // 2 + 1, self.dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight_z = nn.Parameter(torch.randn(8, 8 // 2 + 1, self.dim, 2, dtype=torch.float32) * 0.02)
        self.act = nn.ReLU(inplace=True)
        self.act_2 = nn.ReLU(inplace=True)

        # print(dim)
    def forward(self, x,spatial_size=None):
        # print(f"lens_x : {lens_x}") 256
        z = x[:, 0:64, :]
        x = x[:, 64:, :]
        x = self.norm1(x)
        x1 = x
        # print(f"z.shape{z.shape},x.shape{x.shape}")
        # B, n, C = x.shape
        # if spatial_size is None:
        #     a = b = int(math.sqrt(n))
        # else:
        #     a, b = spatial_size
        # x = x.view(B, a, b, C)
        x = token2patch(x).permute(0,3,2,1)
        B, a, b, C = x.size()
        x = x.to(torch.float32)
        # print(f"x.shape:{x.shape}")
        x_fft = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_fft = x_fft * weight
        x = torch.fft.irfft2(x_fft, s=(a, b), dim=(1, 2), norm='ortho')
        x = patch2token(x.permute(0,3,2,1))
        x = self.norm2(x)
        x = self.mlp(x)

        x1 = token2patch(x1)
        x1 = self.conv3x3(x1)
        x1 = x1 + token2patch(x)
        x1 = patch2token(x1)
        x1 = self.norm3(x1)
        x1 = self.act(x1)

        z = self.norm1(z)
        z1 = z
        z = token2patch(z).permute(0, 3, 2, 1)
        B, t, r, C = z.size()
        z = z.to(torch.float32)
        # print(f"x.shape:{x.shape}")
        z_fft = torch.fft.rfft2(z, dim=(1, 2), norm='ortho')
        weight_z = torch.view_as_complex(self.complex_weight_z)
        z_fft = z_fft * weight_z
        z = torch.fft.irfft2(z_fft, s=(t, r), dim=(1, 2), norm='ortho')
        z = patch2token(z.permute(0, 3, 2, 1))
        z = self.norm2(z)
        z = self.mlp2(z)
        z1 = token2patch(z1)
        z1 = self.conv3x3_2(z1)
        z1 = z1 + token2patch(z)
        z1 = patch2token(z1)
        z1 = self.norm4(z1)
        z1 = self.act_2(z1)
        x = torch.cat([z1, x1], dim=1)

        return x

class Block(nn.Module):
    def __init__(self, dim=768,mlp_ratio=4, drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.down = nn.Linear(dim,self.hidden)
        self.up = nn.Linear(self.hidden,dim)
        self.stb_x1 = SGBlock(dim,mlp_ratio=mlp_ratio)
        self.stb_x2 = SGBlock(dim,mlp_ratio=mlp_ratio)
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)


    def forward(self,x,lens_x):
        res_x = x
        x = self.norm2(self.stb_x(self.norm1(x),lens_x))
        x = res_x + self.mlp(x)
        return x

# fft = SGBlock()
# # print(fft)
# x = torch.ones([2, 320, 768])
# out = fft(x)
# print(out.shape)