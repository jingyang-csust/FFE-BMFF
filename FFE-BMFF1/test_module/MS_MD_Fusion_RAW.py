import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=768, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att2 = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(3)

        xa = x
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xg2 = self.global_att2(xa)
        xlg = xl + xg + xg2
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * x * (1 - wei)
        xo = xo.squeeze(3).permute(0, 2, 1)

        return xo

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        return x
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2)
    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(3)
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        x_output = x * scale
        x_output = x_output.squeeze(3).permute(0, 2, 1)
        return x_output


class MS_MD_Fusion(nn.Module):
    def __init__(self,channels):
        super(MS_MD_Fusion,self).__init__()
        self.aff = AFF(channels=channels)
        self.SG = SpatialGate()


    def forward(self,x):
        xa = self.aff(x=x)
        xb = self.SG(x=x)
        x = x * xa * xb
        return x

x = torch.ones([2,320,768])
channels=x.shape[2]
model = MS_MD_Fusion(channels)
output = model(x)

print(f"6666size:{output.shape}")
