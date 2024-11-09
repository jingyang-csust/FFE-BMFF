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

class SpatialGroupEnhance(nn.Module):
    """
        input:b n c
        output:b n c
    """
    def __init__(self, groups):
        super().__init__()
        self.groups=groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight=nn.Parameter(torch.zeros(1,groups,1,1))
        self.bias=nn.Parameter(torch.zeros(1,groups,1,1))
        self.sig=nn.Sigmoid()
        self.init_weights()


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
        z = x[:, :64, :]
        x = x[:, 64:, :]
        z = token2patch(z)
        x = token2patch(x)

        b, c, h,w=x.shape
        x=x.view(b*self.groups,-1,h,w) #bs*g,dim//g,h,w
        xn=x*self.avg_pool(x) #bs*g,dim//g,h,w
        xn=xn.sum(dim=1,keepdim=True) #bs*g,1,h,w
        t=xn.view(b*self.groups,-1) #bs*g,h*w
        t=t-t.mean(dim=1,keepdim=True) #bs*g,h*w
        std=t.std(dim=1,keepdim=True)+1e-5
        t=t/std #bs*g,h*w
        t=t.view(b,self.groups,h,w) #bs,g,h*w
        t=t*self.weight+self.bias #bs,g,h*w
        t=t.view(b*self.groups,1,h,w) #bs*g,1,h*w
        x=x*self.sig(t)
        x=x.view(b,c,h,w)
        x = patch2token(x)

        b, c, H, W = z.shape
        z = z.view(b * self.groups, -1, H, W)  # bs*g,dim//g,h,w
        zn = z * self.avg_pool(z)  # bs*g,dim//g,h,w
        zn = zn.sum(dim=1, keepdim=True)  # bs*g,1,h,w
        zt = zn.view(b * self.groups, -1)  # bs*g,h*w
        zt = zt - zt.mean(dim=1, keepdim=True)  # bs*g,h*w
        std = zt.std(dim=1, keepdim=True) + 1e-5
        zt = zt / std  # bs*g,h*w
        zt = zt.view(b, self.groups, H, W)  # bs,g,h*w
        zt = zt * self.weight + self.bias  # bs,g,h*w
        zt = zt.view(b * self.groups, 1, H, W)  # bs*g,1,h*w
        z = z * self.sig(zt)
        z = z.view(b, c, H, W)
        z = patch2token(z)

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
        self.sum = SpatialGroupEnhance(groups=8)

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

        x = self.sum(x)

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