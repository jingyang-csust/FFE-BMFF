import torch
from einops import rearrange
from torch import nn
import math

from lib.models.layers.lama_fft1_plus9 import CrossAttention
from lib.models.layers.wave_p import DWT_2D, IDWT_2D  # Assuming wave_p.py contains the wavelet transform code
from lib.utils.token_utils import token2patch, patch2token


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

class DWTBlock(nn.Module):
    def __init__(self, dim, mlp_ratio, h=20, w=20, drop=0., norm_layer=nn.LayerNorm, wavelet='haar'):
        super().__init__()
        mlp_hidden_dim = int(mlp_ratio * dim)
        self.w = w
        self.h = h
        self.norm1 = norm_layer(dim)
        self.dwt = DWT_2D(wavelet)
        self.i_dwt = IDWT_2D(wavelet)
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        # self.basic_conv = nn.Sequential(
        #     nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=1),
        #     nn.BatchNorm2d(dim),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=1),
        #     nn.BatchNorm2d(dim)
        # )
        self.reduce = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True),
        )
        self.filter = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, padding=2, stride=1, groups=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.filter2 = nn.Sequential(
            nn.Conv2d(dim // 4, dim, kernel_size=3, padding=1, stride=1, groups=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        # print(f"res.shape{res.shape}")
        # B, n, C = x.shape
        # if spatial_size is None:
        #     a = b = int(math.sqrt(n))
        # else:
        #     a, b = spatial_size
        # B, a, b, C = x.shape  # ([2, 16, 16, 8])
        x = x.to(torch.float32)
        x_dwt = self.dwt(self.reduce(x))
        # print(x_dwt.shape)

        x_dwt = self.filter(x_dwt)
        # print(x_dwt.shape)

        x_idwt = self.i_dwt(x_dwt)
        # print(x_idwt.shape)

        # print(x_idwt.shape)
        x_idwt = self.filter2(x_idwt)
        # x_idwt = x_idwt.view(B, -1, x_idwt.size(-2) * x_idwt.size(-1)).transpose(1, 2)
        # print(x_idwt.shape)

        return x_idwt

class SGBlock(nn.Module):
    def __init__(self, dim = 8, drop=0.):
        super().__init__()
        self.dim = dim
        # self.adapter_down = nn.Linear(dim, hidden_dim)
        self.stb_x = DWTBlock(self.dim, 4, 16, 16)
        self.stb_z = DWTBlock(self.dim, 4, 8, 8)
        self.norm_x1 = nn.LayerNorm(self.dim)
        self.norm_x2 = nn.LayerNorm(self.dim)
        self.mlp_x = MLP(self.dim,self.dim * 2)
        self.norm_z1 = nn.LayerNorm(self.dim)
        self.norm_z2 = nn.LayerNorm(self.dim)
        self.mlp_z = MLP(self.dim,self.dim * 2)
        # self.mlp2 = Mlp(hidden_dim)
        # self.norm2 = nn.LayerNorm(hidden_dim)
        # self.adapter_up = nn.Linear(hidden_dim, dim)

        # nn.init.zeros_(self.adapter_down.weight)
        # nn.init.zeros_(self.adapter_down.bias)
        # nn.init.zeros_(self.adapter_up.weight)
        # nn.init.zeros_(self.adapter_up.bias)

    def forward(self, x):
        # x = self.adapter_down(x)
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

        z1 = self.norm_z1(patch2token(z))
        z1 = self.stb_z(token2patch(z1))  # ([2, 16, 16, 8])
        z1 = self.norm_z2(patch2token(z1))
        z1 = self.mlp_z(z1)
        zo = z1 + raw_z

        xo_f = torch.cat((zo,xo),dim=1)
        # xo_f = self.adapter_up(xo_f)
        return xo_f

class SG_Block(nn.Module):
    def __init__(self, dim=8, xavier_init=False):
        super().__init__()

        self.adapter_down = nn.Linear(768, dim)
        self.stb1 = SGBlock(dim)
        self.stb2 = SGBlock(dim)
        self.act = nn.ReLU(inplace=True)
        self.norm = nn.LayerNorm(dim)
        self.mlp = MLP(dim,dim * 2)
        self.mlp_x = MLP(dim,dim * 2)
        self.mlp_z = MLP(dim,dim * 2)
        self.attn1 = CrossAttention(dim,dim,dim,dim)
        self.attn2 = CrossAttention(dim,dim,dim,dim)
        self.fnn1 = nn.Linear(dim,dim)
        self.fnn2 = nn.Linear(dim,dim)
        self.adapter_up = nn.Linear(dim, 768)

        #nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        #self.act = QuickGELU()

    def forward(self, x):
        # x_attn = self.shaf(x)

        x_down = self.adapter_down(x)
        raw = x_down
        x_down = self.stb1(x_down)
        x_down = self.stb2(x_down)
        x_down = raw + x_down

        x_down_z = x_down[:,:64,:]
        x_down_x = x_down[:,64:,:]

        x_down_z_raw = x_down_z + self.attn1(self.norm(x_down_z),self.norm(x_down_x))
        x_down_x_raw = x_down_x + self.attn2(self.norm(x_down_x),self.norm(x_down_z))
        x_down_z = self.norm(x_down_z_raw)
        x_down_x = self.norm(x_down_x_raw)
        # x_down_z = x_down_z_raw + self.mlp_z(x_down_z_new)
        # x_down_x = x_down_x_raw + self.mlp_x(x_down_x_new)


        x_down = torch.cat((x_down_z,x_down_x),dim=1)

        x_up = self.adapter_up(x_down)
        # x = x_up + x_attn
        # x = self.norm(self.mlp(x))
        # x = self.mix_atten(x)
        # x = self.mlp2(self.norm2(x))
        # x = x + x_up

        # x = self.sum(x)

        # x_down = self.adapter_down(x)
        # x_down = self.adapter_mid(x_down)
        # x_down = self.dropout(x_down)
        # x_up = self.adapter_up(x_down)
        # # x = x_up + x_attn
        # # x = self.norm(self.mlp(x))
        # x = self.mix_atten(x_up)
        # x = self.mlp2(self.norm2(x))

        return x_up


x = torch.ones(2,320,768)
m = SG_Block()
o = m(x)
print(o.shape)
