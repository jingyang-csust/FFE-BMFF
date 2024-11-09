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

    def forward(self, x,lens_x):
        B, N, C = x.size()
        print(x.shape)

        template = x[:, 0:lens_x, :]
        search = x[:, lens_x:, :]
        B, n, C = search.size()
        print(search.shape)
        H = W = int(n ** 0.5)
        if H * W != n:
            raise ValueError(f"N must be a perfect square, but got {n}")

        # Rearrange input tensors from [B, N, C] to [B, C, H, W]
        x = rearrange(search, 'b (h w) c -> b c h w', h=H, w=W)
        print(search.size)
        # residual = rearrange(residual, 'b (h w) c -> b c h w', h=H, w=W)

        xa = x
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xg2 = self.global_att2(xa)
        xlg = xl + xg + xg2
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * x * (1 - wei)
        xo = rearrange(xo,'b c h w -> b (h w) c')
        x_output = torch.cat([template,xo],dim=1)

        return x_output

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
    def forward(self, x,lens_x):
        B, N, C = x.size()
        template = x[:,:lens_x,:]
        search = x[:,lens_x:,:]
        B,n,C = search.size()
        H = W = int(n ** 0.5)
        if H * W != n:
            raise ValueError(f"N must be a perfect square, but got {n}")
        x = rearrange(search,'b (h w) c -> b h w c', h=H, w=W)
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        x_output = x * scale
        x_output = rearrange(x_output,'b h w c -> b (h w) c')
        x_output = torch.cat([template,x_output],dim=1)
        return x_output


class MS_MD_Fusion(nn.Module):
    def __init__(self,channels):
        super(MS_MD_Fusion,self).__init__()
        self.aff = AFF(channels=channels)
        self.SG = SpatialGate()


    def forward(self,x,lens_x):
        xa = self.aff(x=x)
        xb = self.SG(x=x,lens_x=lens_x)
        x = x * xa * xb
        # print(x.shape)
        return x


if __name__ == '__main__':

    # residual= torch.ones([1,256,768])
    x = torch.ones([2,320,768])
    lens_x = 64
    channels=x.shape[2]
    model = MS_MD_Fusion(channels)

    output = model(x,lens_x)

    print(f"out_put size:{output.shape}")