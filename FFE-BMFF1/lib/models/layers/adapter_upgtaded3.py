import torch
from torch import nn
from torch.nn import init
from lib.models.layers.shaf import Mlp
from lib.utils.token_utils import token2patch, patch2token


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

class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        z = x[:,:64,:]
        x = x[:,64:,:]
        z = token2patch(z)
        x = token2patch(x)

        y=self.gap(x) #bs,c,1,1
        y=y.squeeze(-1).permute(0,2,1) #bs,1,c
        y=self.conv(y) #bs,1,c
        y=self.sigmoid(y) #bs,1,c
        y=y.permute(0,2,1).unsqueeze(-1) #bs,c,1,1
        x = x * y.expand_as(x)
        x = patch2token(x)

        y_z = self.gap(z)  # bs,c,1,1
        y_z = y_z.squeeze(-1).permute(0, 2, 1)  # bs,1,c
        y_z = self.conv(y_z)  # bs,1,c
        y_z = self.sigmoid(y_z)  # bs,1,c
        y_z = y_z.permute(0, 2, 1).unsqueeze(-1)  # bs,c,1,1
        z = z * y_z.expand_as(z)
        z = patch2token((z))
        x = torch.cat((z,x),dim=1)

        return x

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Bi_direct_adapter(nn.Module):
    def __init__(self, dim=8, xavier_init=False):
        super().__init__()

        self.adapter_down = nn.Linear(768, dim)
        self.adapter_up = nn.Linear(dim, 768)
        self.adapter_mid = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(768)
        self.norm2 = nn.LayerNorm(768)
        self.mix_atten = Attention_Module(768)
        self.mlp2 = Mlp(768)
        self.sum = ECAAttention(kernel_size=3)

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
        # ---------------ablation ------------ #
        x = self.sum(x)
        # ---------------ablation ------------ #

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