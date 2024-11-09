import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter

from lib.utils.token_utils import token2patch,patch2token

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

    def __init__(self, channel=768, G=12):
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
        self.linear = nn.Linear(dim * 2, self.channels)

        self.sha_z = ShuffleAttention(dim)
        self.linear_z = nn.Linear(dim * 2, self.channels)

        self.sigmoid = nn.Sigmoid()
        self.mlp1 = Mlp(in_features=self.channels, hidden_features=self.channels * 2, act_layer=nn.GELU)
        self.norm = nn.LayerNorm(self.channels)
        self.mlp1_z = Mlp(in_features=self.channels, hidden_features=self.channels * 2, act_layer=nn.GELU)
        self.norm_z = nn.LayerNorm(self.channels)
    def forward(self,x_v,x_i,lens_z):
        z_v = x_v[:, :lens_z, :]
        x_v = x_v[:, lens_z:, :]
        z_i = x_i[:, :lens_z, :]
        x_i = x_i[:, lens_z:, :]

        z_v = token2patch(z_v)
        x_v = token2patch(x_v)
        z_i = token2patch(z_i)
        x_i = token2patch(x_i)

        x = torch.cat((x_v, x_i), dim=1)  # ([2, 1536, 16, 16])
        x = x.flatten(2).transpose(1, 2).contiguous()  # ([2, 256, 1536])
        # print(x.shape) ([2, 256, 1536])
        x = self.linear(x) # ([2, 256, 768])
        x = token2patch(x) # ([2, 768, 16， 16])
        x = self.sha(x)  # ([2, 768, 16, 16])
        x = self.sigmoid(x)
        x_v = x * x_v  # +
        x_i = x * x_i  # +
        x = x_v + x_i  # *
        x = patch2token(x)
        x = x + self.norm(self.mlp1(x))

        z = torch.cat((z_v, z_i), dim=1)  # ([2, 1536, 16, 16])
        z = z.flatten(2).transpose(1, 2).contiguous()  # ([2, 256, 1536])
        # print(x.shape) ([2, 256, 1536])
        z = self.linear_z(z)  # ([2, 256, 768])
        z = token2patch(z)  # ([2, 768, 16， 16])
        z = self.sha_z(z)  # ([2, 768, 16, 16])
        z = self.sigmoid(z)
        z_v = z * z_v  # +
        z_i = z * z_i  # +
        z = z_v + z_i  # *
        z = patch2token(z)
        z = z + self.norm_z(self.mlp1_z(z))

        x = torch.cat((z,x),dim=1)
        return x
# x = torch.ones(2,320,768)
# m = SHA_Fusion(768)
# o = m(x,x,64)
# print(o.shape)
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)



class MS_Fusion(nn.Module):
    def __init__(self, dim=768 // 2, xavier_init=False):
        super().__init__()

        self.adapter_down = nn.Linear(768, dim)
        self.adapter_mid = SHA_Fusion(dim)
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

    def forward(self, x,xi,lens_x):
        # B, N, C = x.shape
        x_down = self.adapter_down(x)
        xi_down = self.adapter_down(xi)
        x_down = self.adapter_mid(x_down,xi_down,lens_x)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)

        return x_up