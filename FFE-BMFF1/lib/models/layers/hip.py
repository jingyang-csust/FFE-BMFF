import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from lib.utils.token_utils import token2patch, patch2token


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    """
    输入：B C H W
    输出：B C H E
    """
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    """
    输入：B C H W
    输出：B C H E
    """
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class CB11(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pwconv = nn.Conv2d(dim, dim, 1)
        self.bn = nn.BatchNorm2d(dim)

        # Initialize pwconv layer with Kaiming initialization
        init.kaiming_normal_(self.pwconv.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.bn(self.pwconv(x))
        return x.flatten(2).transpose(1, 2).contiguous()

class DWC(nn.Module):
    def __init__(self, dim, kernel):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel, 1, padding='same', groups=dim)

        # Apply Kaiming initialization with fan-in to the dwconv layer
        init.kaiming_normal_(self.dwconv.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2).contiguous()


class LSA(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.pwconv1 = CB11(c2)
        self.dwconv3 = DWC(c2, 3)
        self.dwconv5 = DWC(c2, 5)
        self.dwconv7 = DWC(c2, 7)
        self.pwconv2 = CB11(c2)
        self.fc2 = nn.Linear(c2, c2)

        # Initialize fc1 layer with Kaiming initialization
        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x, H, W) :
        x = self.fc1(x)
        res = x
        x = self.pwconv1(x, H, W)
        x1 = self.dwconv3(x, H, W)
        x2 = self.dwconv5(x, H, W)
        x3 = self.dwconv7(x, H, W)
        return res + self.fc2(F.gelu(self.pwconv2(x + x1 + x2 + x3, H, W)))

class fusion(nn.Module):
    def __init__(self,in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.d_sa1 = SpatialGate()
        self.d_sa2 = SpatialGate()
        self.d_ca1 = ChannelGate(self.in_dim)
        self.d_ca2 = ChannelGate(self.in_dim)
        self.res_block_x_1 = ResBlock(self.in_dim*2,self.in_dim)
        self.res_block_x_2 = ResBlock(self.in_dim)
        self.res_block_z_1 = ResBlock(self.in_dim*2,self.in_dim)
        self.res_block_z_2 = ResBlock(self.in_dim)
    def forward(self,x_v,x_i,lens_z):
        z_v = x_v[:, :lens_z, :]
        x_v = x_v[:, lens_z:, :]
        z_i = x_i[:, :lens_z, :]
        x_i = x_i[:, lens_z:, :]
        z_v = token2patch(z_v)
        x_v = token2patch(x_v)
        z_i = token2patch(z_i)
        x_i = token2patch(x_i)

        x = torch.cat((x_v, x_i), dim=1)
        # B,C,H,W = x.shape
        # x = patch2token(x)
        # x = self.lsa1(x,H,W)
        # x = token2patch(x)
        x = self.res_block_x_1(x)
        res_x = x
        x1 = self.d_sa1(x)
        x2 = self.d_ca1(x)
        x_out = res_x * x1
        x_out = self.res_block_x_2(x_out * x2)
        x_out = patch2token(x_out)

        z = torch.cat((z_v, z_i), dim=1)
        # B, C, h, w = z.shape
        # z = patch2token(z)
        # z = self.lsa2(z, h, w)
        # z = token2patch(z)
        z = self.res_block_z_1(z)
        res_z = z
        z1 = self.d_sa2(z)
        z2 = self.d_ca2(z)
        z_out = res_z * z1
        z_out = self.res_block_z_2(z_out * z2)
        z_out = patch2token(z_out)

        x_f = torch.cat((z_out,x_out),dim=1)
        return x_f

class MS_Fusion(nn.Module):
    def __init__(self, dim=256, xavier_init=False):
        super().__init__()

        self.adapter_down = nn.Linear(768, dim)
        self.adapter_mid = fusion(dim)
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


# x = torch.ones(2,320,768)
# m = MS_Fusion()
# o = m(x,x,64)
# print(x.shape)

