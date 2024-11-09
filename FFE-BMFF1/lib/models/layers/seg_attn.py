import torch
from torch import nn

from lib.utils.token_utils import token2patch, patch2token


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim),
            nn.Conv2d(dim, dim, kernel_size=1)  # Pointwise Convolution
        )
        self.conv0_2 = nn.Sequential(
            nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim),
            nn.Conv2d(dim, dim, kernel_size=1)  # Pointwise Convolution
        )

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim),
            nn.Conv2d(dim, dim, kernel_size=1)  # Pointwise Convolution
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim),
            nn.Conv2d(dim, dim, kernel_size=1)  # Pointwise Convolution
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim),
            nn.Conv2d(dim, dim, kernel_size=1)  # Pointwise Convolution
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim),
            nn.Conv2d(dim, dim, kernel_size=1)  # Pointwise Convolution
        )
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):
    """
    输入： B C H W
    输出： B C H W
    """
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        # B, N, C = x.shape
        # x = x.permute(0, 2, 1).view(B, C, H, W)
        # print(self.attn(self.norm1(x)).shape) ([2, 768, 16, 16])
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                               * self.attn(self.norm1(x)))
        # print(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1).shape) ([768, 1, 1])
        # print(self.norm2(x).shape) ([2, 768, 16, 16])
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                               * token2patch(self.mlp(patch2token(self.norm2(x)))))
        # x = x.view(B, C, N).permute(0, 2, 1)
        return x

class fuision(nn.Module):
    """
    输入；B N C
    输出：B N C
    """
    def __init__(self,dim):
        super().__init__()
        self.block1 = Block(dim)
        self.block2 = Block(dim)

    def forward(self,x_v,x_i,lens_z):
        z_v = x_v[:, :lens_z, :]
        x_v = x_v[:, lens_z:, :]
        z_i = x_i[:, :lens_z, :]
        x_i = x_i[:, lens_z:, :]
        z_v = token2patch(z_v)
        x_v = token2patch(x_v)
        z_i = token2patch(z_i)
        x_i = token2patch(x_i)

        x = x_v + x_i
        x = self.block1(x)
        x = patch2token(x)

        z = z_v + z_i
        z = self.block2(z)
        z = patch2token(z)

        x = torch.cat((z,x),dim=1)
        return x

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)



class MS_Fusion(nn.Module):
    def __init__(self, dim=8, upscale_dim=1024, xavier_init=False):
        super().__init__()

        self.adapter_down = nn.Linear(768, dim)
        self.adapter_mid = fuision(dim)
        self.adapter_up = nn.Linear(dim, 768)

        #nn.init.xavier_uniform_(self.adapter_down.weight)
        # nn.init.zeros_(self.adapter_mid.bias)
        # nn.init.zeros_(self.adapter_mid.weight)

        # nn.init.zeros_(self.adapter_mid_upscale.bias)
        # nn.init.zeros_(self.adapter_mid_upscale.weight)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)

        #self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim
        self.upscale_dim = upscale_dim

    def forward(self, x,xi,lens_x):
        # B, N, C = x.shape
        x_down = self.adapter_down(x)
        xi_down = self.adapter_down(xi)
        x_down = self.adapter_mid(x_down,xi_down,lens_x)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)

        return x_up

# x = torch.ones(2,320,768)
# m = fuision(768)
# o = m(x,x,64)
# print(o.shape)