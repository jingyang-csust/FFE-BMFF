import torch
from torch import nn
from torch.nn import init, Parameter
from lib.models.layers.shaf import Mlp
from lib.utils.token_utils import token2patch, patch2token
import torch.nn.functional as F

class MultiScaleDWConv(nn.Module):
    def __init__(self, dim, scale=(1, 3, 5, 7)):
        super().__init__()
        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)
            conv = nn.Conv2d(channels, channels,
                             kernel_size=scale[i],
                             padding=scale[i] // 2,
                             groups=channels)
            self.channels.append(channels)
            self.proj.append(conv)

    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x


class Mlp_plus(nn.Module):  ### MS-FFN
    """
    Mlp implemented by with 1x1 convolutions.

    Input: Tensor with shape [B, C, H, W].
    Output: Tensor with shape [B, C, H, W].
    Args:
        in_features (int): Dimension of input features.
        hidden_features (int): Dimension of hidden features.
        out_features (int): Dimension of output features.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        drop (float): Dropout rate. Defaults to 0.0.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 drop=0, ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(hidden_features),
        )
        self.dwconv = MultiScaleDWConv(hidden_features)
        self.act = nn.GELU()
        self.norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, in_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_features),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        z_v = x[:, :64, :]
        x_v = x[:, 64:, :]

        z_v = token2patch(z_v)
        x_v = token2patch(x_v)

        x_v = self.fc1(x_v)
        x_v = self.dwconv(x_v) + x_v
        x_v = self.norm(self.act(x_v))
        x_v = self.drop(x_v)
        x_v = self.fc2(x_v)
        x_v = self.drop(x_v)
        x_v = patch2token(x_v)

        z_v = self.fc1(z_v)
        z_v = self.dwconv(z_v) + z_v
        z_v = self.norm(self.act(z_v))
        z_v = self.drop(z_v)
        z_v = self.fc2(z_v)
        z_v = self.drop(z_v)
        z_v = patch2token(z_v)

        x = torch.cat((z_v,x_v),dim=1)
        return x

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

class ShuffleAttention(nn.Module):

    def __init__(self, channel=8, G=1):
        super().__init__()
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()

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

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # 扁平化
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.size()
        # 将通道分成子特征
        x = x.view(b * self.G, -1, h, w)  # bs*G,c//G,h,w

        # 通道分割
        x_0, x_1 = x.chunk(2, dim=1)  # bs*G,c//(2*G),h,w

        # 通道注意力
        x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1,1
        x_channel = self.cweight * x_channel + self.cbias  # bs*G,c//(2*G),1,1
        x_channel = x_0 * self.sigmoid(x_channel)

        # 空间注意力
        x_spatial = self.gn(x_1)  # bs*G,c//(2*G),h,w
        x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),h,w
        x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),h,w

        # 沿通道轴拼接
        out = torch.cat([x_channel, x_spatial], dim=1)  # bs*G,c//G,h,w
        out = out.contiguous().view(b, -1, h, w)

        # 通道混洗
        out = self.channel_shuffle(out, 2)
        return out

class SHA_Fusion(nn.Module):
    """
    BNC
    BNC
    """
    def __init__(self,dim):
        super().__init__()
        self.channels = dim

        self.sha = ShuffleAttention(dim)

        self.sha_z = ShuffleAttention(dim)

    def forward(self,x):
        z_v = x[:, :64, :]
        x_v = x[:, 64:, :]

        z_v = token2patch(z_v)
        x_v = token2patch(x_v)

        x = self.sha(x_v)  # ([2, 768, 16, 16])
        x = patch2token(x)

        z = self.sha_z(z_v)  # ([2, 768, 16, 16])
        z = patch2token(z)

        x = torch.cat((z,x),dim=1)
        return x

class Attention_Module(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.linear = nn.Linear(dim, 8)
        self.act1 = nn.ReLU(inplace=True)
        self.cross_attn = SHA_Fusion(8)
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
        self.shaf = Attention_Module(768)
        self.mlp2 = Mlp_plus(768)
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
        # x = self.mix_atten(x)
        # x = self.mlp2(self.norm2(x))
        # x = x + x_up
        x = self.shaf(x)
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

# x = torch.ones(16,320,768)
# m = Bi_direct_adapter(768)
# o = m(x)
# print(o.shape)