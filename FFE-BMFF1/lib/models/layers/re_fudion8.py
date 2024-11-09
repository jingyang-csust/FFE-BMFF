from functools import partial

import torch
from timm.layers import to_2tuple
from torch import nn
from lib.models.layers.shaf import ShuffleAttention
from lib.utils.token_utils import token2patch, patch2token
from test_module.dwconv import MultiScaleDWConv
from torch.nn import functional as F

class DynamicConv2d(nn.Module):  ### IDConv
    def __init__(self,
                 dim,
                 kernel_size=3,
                 reduction_ratio=4,
                 num_groups=8,
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



class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class fuision(nn.Module):
    """

    输入；B N C
    输出：B N C
    """
    def __init__(self,dim):
        super().__init__()
        self.channels = dim
        self.block1 = ShuffleAttention(self.channels)
        self.linear = nn.Linear(dim * 2, self.channels)
        self.block2 = DynamicConv2d(self.channels,num_groups=2)


    def forward(self,x1,x2):

        x = torch.cat((x1, x2), dim=1)  # ([2, 1536, 16, 16])
        x = x.flatten(2).transpose(1, 2).contiguous()  # ([2, 256, 1536])
        x = self.linear(x)  # ([2, 256, 768])
        x = token2patch(x)  # ([2, 768, 16， 16])
        x1 = self.block1(x)
        x2 = self.block2(x)
        xo_tmp = x1 + x2
        x_v = xo_tmp + x1
        x_i = xo_tmp + x2



        return x_v,x_i

class Unify(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.channels = dim
        self.mlp1 = Mlp(in_features=self.channels, hidden_features=self.channels * 2)
        self.norm = nn.LayerNorm(self.channels)

    def forward(self,x_v,x_i):
        B, C, H, W = x_v.shape
        N = int(H * W)
        x_v = patch2token(x_v)
        x_i = patch2token(x_i)
        x = torch.cat((x_v, x_i), dim=1)
        x = x + self.norm(self.mlp1(x))
        x_v, x_i = torch.split(x, (N, N,), dim=1)

        x_v = token2patch(x_v)
        x_i = token2patch(x_i)

        return x_v,x_i

class MS_Fusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fusion = fuision(dim)
        self.unify1 = Unify(dim)
        self.unify2 = Unify(dim)
    def forward(self, x,xi,lens_z):
        z_v = x[:, :lens_z, :]
        x_v = x[:, lens_z:, :]
        z_i = xi[:, :lens_z, :]
        x_i = xi[:, lens_z:, :]

        z_v = token2patch(z_v)
        x_v = token2patch(x_v)
        z_i = token2patch(z_i)
        x_i = token2patch(x_i)


        z_v, z_i = self.fusion(z_v, z_i)
        x_v, x_i = self.fusion(x_v, x_i)

        z_v, z_i = self.unify1(z_v, z_i)
        x_v, x_i = self.unify2(x_v, x_i)

        z_v = patch2token(z_v)
        x_v = patch2token(x_v)
        z_i = patch2token(z_i)
        x_i = patch2token(x_i)

        x_v = torch.cat((z_v, x_v), dim=1)
        x_i = torch.cat((z_i, x_i), dim=1)

        return x_v + x,x_i + xi



# x = torch.ones(2,320,768)
# m = MS_Fusion(768)
# o,oi = m(x,x,64)
