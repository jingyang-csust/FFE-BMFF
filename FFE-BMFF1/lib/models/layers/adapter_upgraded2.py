# import torch
# from torch import nn
# import torch.nn.functional as F
# from torch.nn import init
#
# from lib.models.layers.CFN import CFN
# from lib.models.layers.shaf import ShuffleAttention, Mlp
# from lib.models.layers.sin_shaf_advan import SHA_Fusion
# from lib.models.layers.transnext import ConvolutionalGLU1
# from lib.utils.token_utils import token2patch, patch2token
#
# class LPU(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
#
#     def forward(self, x):
#         return self.conv(x) + x
#
# class CB11(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.pwconv = nn.Conv2d(dim, dim, 1)
#         self.bn = nn.BatchNorm2d(dim)
#
#         # Initialize pwconv layer with Kaiming initialization
#         init.kaiming_normal_(self.pwconv.weight, mode='fan_in', nonlinearity='relu')
#
#     def forward(self, x, H, W):
#         B, _, C = x.shape
#         x = x.transpose(1, 2).view(B, C, H, W).contiguous()
#         x = self.bn(self.pwconv(x))
#         return x.flatten(2).transpose(1, 2).contiguous()
#
# class DWC(nn.Module):
#     def __init__(self, dim, kernel):
#         super().__init__()
#         self.dwconv = nn.Conv2d(dim, dim, kernel, 1, padding='same', groups=dim)
#
#         # Apply Kaiming initialization with fan-in to the dwconv layer
#         init.kaiming_normal_(self.dwconv.weight, mode='fan_in', nonlinearity='relu')
#
#     def forward(self, x, H, W):
#         B, _, C = x.shape
#         x = x.transpose(1, 2).view(B, C, H, W).contiguous()
#         x = self.dwconv(x)
#         return x.flatten(2).transpose(1, 2).contiguous()
#
# class LSA(nn.Module):
#     def __init__(self, c1, c2):
#         super().__init__()
#         self.fc1 = nn.Linear(c1, c2)
#         self.pwconv1 = CB11(c2)
#         self.dwconv3 = DWC(c2, 3)
#         self.dwconv5 = DWC(c2, 5)
#         self.dwconv7 = DWC(c2, 7)
#         self.pwconv2 = CB11(c2)
#         self.fc2 = nn.Linear(c2, c1)
#
#         # Initialize fc1 layer with Kaiming initialization
#         init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
#         init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
#
#     def forward(self, x, H, W) :
#         x = self.fc1(x)
#         x = self.pwconv1(x, H, W)
#         x1 = self.dwconv3(x, H, W)
#         x2 = self.dwconv5(x, H, W)
#         x3 = self.dwconv7(x, H, W)
#         return self.fc2(F.gelu(self.pwconv2(x + x1 + x2 + x3, H, W)))
#
# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         assert dim % num_heads == 0, 'dim should be divisible by num_heads'
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#         self.qkv_mem = None
#
#     def forward(self, x):
#         """
#         x is a concatenated vector of template and search region features.
#         """
#         t_h = 8
#         t_w = 8
#         s_h = 16
#         s_w = 16
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
#         q_mt, q_s = torch.split(q, [t_h * t_w, s_h * s_w], dim=2)
#         k_mt, k_s = torch.split(k, [t_h * t_w, s_h * s_w], dim=2)
#         v_mt, v_s = torch.split(v, [t_h * t_w, s_h * s_w], dim=2)
#
#         # asymmetric mixed attention
#         attn = (q_mt @ k_mt.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#         x_mt = (attn @ v_mt).transpose(1, 2).reshape(B, t_h*t_w, C)
#
#         attn = (q_s @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#         x_s = (attn @ v).transpose(1, 2).reshape(B, s_h*s_w, C)
#
#         x = torch.cat([x_mt, x_s], dim=1)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
#
# class Attention_Module(nn.Module):
#     def __init__(self, dim, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.linear = nn.Linear(dim, dim // 2)
#         self.act1 = nn.ReLU(inplace=True)
#         self.cross_attn = Attention( dim // 2, num_heads=8)
#         self.end_proj = nn.Linear( dim // 2, dim)
#         self.norm1 = norm_layer(dim)
#
#     def forward(self, x1):
#         # y1, u1 = self.act1(self.linear(x1))
#         # y2, u2 = self.act2(self.linear(x2))
#         # v1, v2 = self.cross_attn(u1, u2)
#         y = self.act1(self.linear(x1))
#         v = self.cross_attn(y)
#
#         y1 = y + v
#
#         out_x1 = self.norm1(x1 + self.end_proj(y1))
#         return out_x1
#
# class QuickGELU(nn.Module):
#     def forward(self, x: torch.Tensor):
#         return x * torch.sigmoid(1.702 * x)
#
#
# class Bi_direct_adapter(nn.Module):
#     def __init__(self, dim=8, xavier_init=False):
#         super().__init__()
#
#         self.adapter_down = nn.Linear(768, dim)
#         self.adapter_up = nn.Linear(dim, 768)
#         self.adapter_mid = nn.Linear(dim, dim)
#         self.act1 = nn.ReLU(inplace=True)
#         self.mix_attn = Attention(dim, num_heads=8)
#         self.end_proj = nn.Linear( dim // 2, dim)
#         self.norm1 = nn.LayerNorm(dim)
#         # self.shaf = SHA_Fusion(768)
#         # self.norm = nn.LayerNorm(768)
#         # self.norm2 = nn.LayerNorm(768)
#         # self.mlp = ConvolutionalGLU1()
#         # self.mix_atten = Attention_Module(768)
#         # self.mlp2 = Mlp(768)
#         # self.cfn = CFN(768,768)
#         # self.lpu = LPU(768)
#         # self.lsa = LSA(768,8)
#         #nn.init.xavier_uniform_(self.adapter_down.weight)
#         nn.init.zeros_(self.adapter_mid.bias)
#         nn.init.zeros_(self.adapter_mid.weight)
#         nn.init.zeros_(self.adapter_down.weight)
#         nn.init.zeros_(self.adapter_down.bias)
#         nn.init.zeros_(self.adapter_up.weight)
#         nn.init.zeros_(self.adapter_up.bias)
#
#         #self.act = QuickGELU()
#         self.dropout = nn.Dropout(0.1)
#         self.dim = dim
#
#     def forward(self, x):
#         # x_attn = self.shaf(x)
#
#         # mixform1
#         x_down = self.adapter_down(x)
#
#         x_pre_attn = self.act1(x_down)
#         x_attn = self.mix_attn(x_pre_attn)
#         x_attn = x_pre_attn + x_attn
#
#         x_down = self.adapter_mid(x_down)
#         x_down = x_down + x_attn
#
#         x_down = self.dropout(x_down)
#         x_up = self.adapter_up(x_down)
#         # x = x_up + x_attn
#         # x = self.norm(self.mlp(x))
#
#         # x_ = x[:,:64,:]
#         # z_ = x[:,64:,:]
#         # x_ = patch2token(self.lpu(token2patch(x_)))
#         # z_ = patch2token(self.lpu(token2patch(z_)))
#         # x_ = torch.cat((z_,x_),dim=1)
#
#         # x = self.mix_atten(x)
#         # x = self.mlp2(self.norm2(x))
#         # x = x + x_up
#
#         # x = x + x_up
#
#
#
#
#         # x_down = self.adapter_down(x)
#         # x_down = self.adapter_mid(x_down)
#         # x_down = self.dropout(x_down)
#         # x_up = self.adapter_up(x_down)
#         #
#         # x_up1 = x[:,:64,:]
#         # x_up2 = x[:,64:,:]
#         # x_up1 = self.lsa(x_up1,8,8)
#         # x_up2 = self.lsa(x_up2,16,16)
#         # x_sum = torch.cat((x_up2,x_up1),dim=1)
#         #
#         # x_up = x_sum + x_up
#         # # x = x_up + x_attn
#         # # x = self.norm(self.mlp(x))
#         # x_1 = self.mix_atten(x_up)
#         # x = self.mlp2(self.norm2(x_1))
#         # x = x + x_up
#
#         return x_up
#
# # x = torch.ones(2,320,768)
# # m = Bi_direct_adapter(768)
# # o = m(x)
# # print(o.shape)


import torch
from torch import nn

from lib.models.layers.shaf import ShuffleAttention, Mlp
from lib.models.layers.sin_shaf_advan import SHA_Fusion
from lib.models.layers.transnext import ConvolutionalGLU1

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qkv_mem = None

    def forward(self, x):
        """
        x is a concatenated vector of template and search region features.
        """
        t_h = 8
        t_w = 8
        s_h = 16
        s_w = 16
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        q_mt, q_s = torch.split(q, [t_h * t_w, s_h * s_w], dim=2)
        k_mt, k_s = torch.split(k, [t_h * t_w, s_h * s_w], dim=2)
        v_mt, v_s = torch.split(v, [t_h * t_w, s_h * s_w], dim=2)

        # asymmetric mixed attention
        attn = (q_mt @ k_mt.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_mt = (attn @ v_mt).transpose(1, 2).reshape(B, t_h*t_w, C)

        attn = (q_s @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_s = (attn @ v).transpose(1, 2).reshape(B, s_h*s_w, C)

        x = torch.cat([x_mt, x_s], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention_Module(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.linear = nn.Linear(dim, 8)
        self.act1 = nn.ReLU(inplace=True)
        self.cross_attn = Attention(8, num_heads=8)
        self.end_proj = nn.Linear(8, dim)
        self.norm1 = norm_layer(dim)

    def forward(self, x1):
        # y1, u1 = self.act1(self.linear(x1))
        # y2, u2 = self.act2(self.linear(x2))
        # v1, v2 = self.cross_attn(u1, u2)
        y = self.act1(self.linear(x1))
        v = self.cross_attn(y)

        y1 = y + v

        out_x1 = self.norm1(x1 + self.end_proj(y1))
        return out_x1

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Bi_direct_adapter(nn.Module):
    def __init__(self, dim=8, xavier_init=False):
        super().__init__()

        self.adapter_down = nn.Linear(768, dim)
        self.adapter_up = nn.Linear(dim, 768)
        self.adapter_mid = nn.Linear(dim, dim)
        self.shaf = SHA_Fusion(768)
        self.norm = nn.LayerNorm(768)
        self.norm2 = nn.LayerNorm(768)
        self.mlp = ConvolutionalGLU1()
        self.mix_atten = Attention_Module(768)
        self.mlp2 = Mlp(768)

        #nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_mid.bias)
        nn.init.zeros_(self.adapter_mid.weight)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        #self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        # x_attn = self.shaf(x)

        # mixform1
        x_down = self.adapter_down(x)
        x_down = self.adapter_mid(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)
        # x = x_up + x_attn
        # x = self.norm(self.mlp(x))
        x = self.mix_atten(x)
        x = self.mlp2(self.norm2(x))
        x = x + x_up

        # x_down = self.adapter_down(x)
        # x_down = self.adapter_mid(x_down)
        # x_down = self.dropout(x_down)
        # x_up = self.adapter_up(x_down)
        # # x = x_up + x_attn
        # # x = self.norm(self.mlp(x))
        # x = self.mix_atten(x_up)
        # x = self.mlp2(self.norm2(x))

        return x

# x = torch.ones(2,320,768)
# m = Bi_direct_adapter(768)
# o = m(x)
# print(o.shape)

