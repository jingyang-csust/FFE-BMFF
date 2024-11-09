import torch
from torch import nn
from lib.models.layers.epsb import PSAModule
from lib.models.layers.las import LSA
from lib.models.layers.shaf import ShuffleAttention
from lib.utils.token_utils import token2patch, patch2token
from test_module.dwconv import MultiScaleDWConv
from torch.nn import functional as F

class Attention(nn.Module):  ### OSRA
    def __init__(self, dim,
                 num_heads=1,
                 qk_scale=None,
                 attn_drop=0,
                 sr_ratio=1, ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio
        self.q = nn.Conv2d(dim, dim, kernel_size=1)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1)
        self.attn_drop = nn.Dropout(attn_drop)
        if sr_ratio > 1:
            self.sr = nn.Sequential(
                nn.Conv2d(dim, dim,
                           kernel_size=sr_ratio + 3,
                           stride=sr_ratio,
                           padding=(sr_ratio + 3) // 2,
                           groups=dim,
                           bias=False),
                nn.BatchNorm2d(dim),
                nn.GELU(),
                nn.Conv2d(dim, dim,
                           kernel_size=1,
                           groups=dim,
                           bias=False),
                nn.BatchNorm2d(dim),
                nn.GELU()
            )
        else:
            self.sr = nn.Identity()
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x, relative_pos_enc=None):
        B, C, H, W = x.shape
        q = self.q(x).reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)
        kv = self.sr(x)
        kv = self.local_conv(kv) + kv
        k, v = torch.chunk(self.kv(kv), chunks=2, dim=1)
        k = k.reshape(B, self.num_heads, C // self.num_heads, -1)
        v = v.reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)
        attn = (q @ k) * self.scale
        if relative_pos_enc is not None:
            if attn.shape[2:] != relative_pos_enc.shape[2:]:
                relative_pos_enc = F.interpolate(relative_pos_enc, size=attn.shape[2:],
                                                 mode='bicubic', align_corners=False)
            attn = attn + relative_pos_enc
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(-1, -2)
        return x.reshape(B, C, H, W)

class DynamicConv2d(nn.Module):  ### IDConv
    def __init__(self,
                 dim,
                 kernel_size=3,
                 reduction_ratio=4,
                 num_groups=1,
                 bias=True):
        super().__init__()
        assert num_groups > 1, f"num_groups {num_groups} should > 1."
        self.num_groups = num_groups
        self.K = kernel_size
        self.bias_type = bias
        self.weight = nn.Parameter(torch.empty(num_groups, dim, kernel_size, kernel_size), requires_grad=True)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(kernel_size, kernel_size))
        self.proj = nn.Sequential(
            nn.Conv2d(dim,
                       dim // reduction_ratio,
                       kernel_size=1),
            nn.BatchNorm2d(dim // reduction_ratio),
            nn.GELU(),
            nn.Conv2d(dim // reduction_ratio, dim * num_groups, kernel_size=1)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(num_groups, dim), requires_grad=True)
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x):

        B, C, H, W = x.shape
        scale = self.proj(self.pool(x)).reshape(B, self.num_groups, C, self.K, self.K)
        scale = torch.softmax(scale, dim=1)
        weight = scale * self.weight.unsqueeze(0)
        weight = torch.sum(weight, dim=1, keepdim=False)
        weight = weight.reshape(-1, 1, self.K, self.K)

        if self.bias is not None:
            scale = self.proj(torch.mean(x, dim=[-2, -1], keepdim=True))
            scale = torch.softmax(scale.reshape(B, self.num_groups, C), dim=1)
            bias = scale * self.bias.unsqueeze(0)
            bias = torch.sum(bias, dim=1).flatten(0)
        else:
            bias = None

        x = F.conv2d(x.reshape(1, -1, H, W),
                     weight=weight,
                     padding=self.K // 2,
                     groups=B * C,
                     bias=bias)

        return x.reshape(B, C, H, W)



class Mlp(nn.Module):  ### MS-FFN
    """
    Input: Tensor with shape [B, C, H, W].
    Output: Tensor with shape [B, C, H, W].
    """

    def __init__(self,
                 in_features,
                 h,
                 w,
                 hidden_features=None,
                 out_features=None,
                 drop=0, ):
        super().__init__()
        self.h = h
        self.w = w
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(hidden_features),
        )
        self.dwconv = MultiScaleDWConv(hidden_features,self.h,self.w)
        self.act = nn.GELU()
        self.norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, in_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_features),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)

        x = self.dwconv(x) + x
        x = self.norm(self.act(x))

        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x
class fuision(nn.Module):
    """
    输入；B N C
    输出：B N C
    """
    def __init__(self,dim):
        super().__init__()
        self.channels = dim
        self.block1_x = ShuffleAttention(self.channels)
        self.linear_x = nn.Linear(dim * 2, self.channels)
        self.block2_x = Attention(self.channels)
        self.block1_z = ShuffleAttention(self.channels)
        self.linear_z = nn.Linear(dim * 2, self.channels)
        self.block2_z = Attention(self.channels)

        self.mlp1 = Mlp(in_features=self.channels, hidden_features=self.channels * 2,out_features=self.channels,h=16,w=16)
        self.mlp2 = Mlp(in_features=self.channels, hidden_features=self.channels * 2,out_features=self.channels,h=8,w=8)
        self.norm = nn.LayerNorm(self.channels)

    def forward(self,x_v,x_i,lens_z):
        z_v = x_v[:, :lens_z, :]
        x_v = x_v[:, lens_z:, :]
        z_i = x_i[:, :lens_z, :]
        x_i = x_i[:, lens_z:, :]

        z_v = token2patch(z_v)
        x_v = token2patch(x_v)
        z_i = token2patch(z_i)
        x_i = token2patch(x_i)

        # x = x_v + x_i
        x = torch.cat((x_v, x_i), dim=1)  # ([2, 1536, 16, 16])
        x = x.flatten(2).transpose(1, 2).contiguous()  # ([2, 256, 1536])
        # print(x.shape) ([2, 256, 1536])
        x = self.linear_x(x)  # ([2, 256, 768])
        x = token2patch(x)  # ([2, 768, 16， 16])
        x1 = self.block1_x(x)
        x2 = self.block2_x(x)
        xo_tmp = x1 + x2
        xo1 = xo_tmp * x_v
        xo2 = xo_tmp * x_i
        # xo = torch.cat((xo1, xo2), dim=1)  # ([2, 1536, 16, 16])
        # xo = xo.flatten(2).transpose(1, 2).contiguous()  # ([2, 256, 1536])
        xo = xo1 + xo2
        # xo = patch2token(xo)
        xo = xo + self.mlp1(token2patch(self.norm(patch2token(xo))))
        xo = patch2token(xo)

        # z = z_v + z_i
        z = torch.cat((z_v, z_i), dim=1)  # ([2, 1536, 16, 16])
        z = z.flatten(2).transpose(1, 2).contiguous()  # ([2, 256, 1536])
        # print(x.shape) ([2, 256, 1536])
        z = self.linear_z(z)  # ([2, 256, 768])
        z = token2patch(z)  # ([2, 768, 16， 16])
        z1 = self.block1_z(z)
        z2 = self.block2_z(z)
        zo_tmp = z1 + z2
        zo1 = zo_tmp * z_v
        zo2 = zo_tmp * z_i
        # zo = torch.cat((zo1, zo2), dim=1)  # ([2, 1536, 16, 16])
        zo = zo1 + zo2
        # zo = zo.flatten(2).transpose(1, 2).contiguous()  # ([2, 256, 1536])
        zo = zo + self.mlp2(token2patch(self.norm(patch2token(zo))))
        zo = patch2token(zo)

        x = torch.cat((zo,xo),dim=1)
        return x

class MS_Fusion(nn.Module):
    def __init__(self, dim=768 // 2, xavier_init=False):
        super().__init__()

        self.adapter_down = nn.Linear(768, dim)
        self.adapter_mid = fuision(dim)
        self.adapter_up = nn.Linear(dim, 768)

        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)

        #self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x,xi,lens_x):
        # B, N, C = x.shape
        x_down = self.adapter_down(x)
        xi_down = self.adapter_down(xi)
        x_down = self.adapter_mid(x_down,xi_down,lens_x)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)

        return x_up
    
# x = torch.ones(2,320,768)
# m = MS_Fusion()
# o = m(x,x,64)
# print(o.shape)