# from timm.models.vision_transformer import Attention
import torch.nn.functional as F
import torch
from lib.models.layers.adapter_upgtaded3 import Attention
from timm.layers import DropPath
from torch import nn
from lib.utils.token_utils import token2patch,patch2token


class CrossAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim, v_dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim * num_heads, in_dim1)

    def forward(self, x1, x2, mask=None):
        batch_size, seq_len1, in_dim1 = x1.size()
        seq_len2 = x2.size(1)

        q1 = self.proj_q1(x1).view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k2 = self.proj_k2(x2).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        v2 = self.proj_v2(x2).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q1, k2) / self.k_dim ** 0.5

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1)
        output = self.proj_o(output)

        return output

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
        # self.norm1 = nn.BatchNorm2d(dim)
        # self.norm2 = nn.BatchNorm2d(dim)
        # Initialize complex weights
        self.complex_weight = nn.Parameter(torch.randn(h, w // 2 + 1, dim, 2, dtype=torch.float32) * 0.02)
        # Initialize 1x1 convolution and GELU activation
        # self.conv1x1_real = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        # self.conv1x1_imag = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
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
    def __init__(self, dim = 8, drop=0.):
        super().__init__()
        self.dim = dim
        # self.adapter_down = nn.Linear(dim, hidden_dim)
        self.stb_x = SpectralGatingNetwork(self.dim, 16, 16)
        self.stb_z = SpectralGatingNetwork(self.dim, 8, 8)
        self.norm_x1 = nn.LayerNorm(self.dim)
        self.norm_x2 = nn.LayerNorm(self.dim)
        self.mlp_x = FeedForward(self.dim,self.dim * 2)
        self.norm_z1 = nn.LayerNorm(self.dim)
        self.norm_z2 = nn.LayerNorm(self.dim)
        self.mlp_z = FeedForward(self.dim,self.dim * 2)
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
        # self.act = nn.ReLU(inplace=True)
        self.norm = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim,dim * 2)
        self.attn1 = Attention(dim)
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

        x_down = x_down + self.attn1(self.norm(x_down))
        x_down_raw = self.norm(x_down)
        x_down = x_down + self.mlp(x_down_raw)

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
# x = torch.ones(2,320,768)
# m = SG_Block(768)
# o = m(x)
# print(o.shape)