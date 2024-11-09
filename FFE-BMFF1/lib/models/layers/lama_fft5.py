import torch
from torch import nn
import torch.nn.functional as F

from lib.models.layers.adapter_upgtaded3 import ECAAttention
from lib.utils.token_utils import token2patch,patch2token
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        res = x * y.expand_as(x)
        return res
class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

        # squeeze and excitation block
        self.use_se = use_se
        if use_se:
            if se_kwargs is None:
                se_kwargs = {}
            self.se = SELayer(self.conv_layer.in_channels, **se_kwargs)

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode, align_corners=False)

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        if self.use_se:
            ffted = self.se(ffted)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        return output

class SpectralTransform(nn.Module):

    def __init__(self, dim = 8, stride=1, groups=1, enable_lfu=True, **fu_kwargs):
        # bn_layer not used
        self.hidden_dim = dim
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
            self.downsample_z = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()
            self.downsample_z = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(self.hidden_dim // 2),
            nn.ReLU(inplace=True)
        )
        self.conv1_z = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(self.hidden_dim // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            self.hidden_dim // 2, self.hidden_dim // 2, groups, **fu_kwargs)
        self.fu_z = FourierUnit(
            self.hidden_dim // 2, self.hidden_dim // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                self.hidden_dim // 2, self.hidden_dim // 2, groups)
            self.lfu_z = FourierUnit(
                self.hidden_dim // 2, self.hidden_dim // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            self.hidden_dim // 2, self.hidden_dim, kernel_size=1, groups=groups, bias=False)
        self.conv2_z = torch.nn.Conv2d(
            self.hidden_dim // 2, self.hidden_dim, kernel_size=1, groups=groups, bias=False)

        self.conv3x3 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv3x3_2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm2d(self.hidden_dim)
        self.norm_2 = nn.BatchNorm2d(self.hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.act_2 = nn.ReLU(inplace=True)
    def forward(self, x):
        z = x[:,:64,:]
        x = x[:,64:,:]
        x = token2patch(x)
        z = token2patch(z)

        x1 = x
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)
        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0
        output = self.conv2(x + output + xs)
        output = patch2token(output)
        x1 = self.conv3x3(x1)
        x1 = x1 + token2patch(output)
        x1 = self.norm(x1)
        x1 = patch2token(x1)
        x1 = self.act(x1)

        z1 = z
        z = self.downsample_z(z)
        z = self.conv1_z(z)
        output_z = self.fu_z(z)
        if self.enable_lfu:
            n, c, h, w = z.shape
            split_no = 2
            split_s_z = h // split_no
            zs = torch.cat(torch.split(
                z[:, :c // 4], split_s_z, dim=-2), dim=1).contiguous()
            zs = torch.cat(torch.split(zs, split_s_z, dim=-1),
                           dim=1).contiguous()
            zs = self.lfu_z(zs)
            zs = zs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            zs = 0
        output_z = self.conv2_z(z + output_z + zs)
        output_z = patch2token(output_z)
        z1 = self.conv3x3_2(z1)
        z1 = z1 + token2patch(output_z)
        z1 = self.norm_2(z1)
        z1 = patch2token(z1)
        z1 = self.act_2(z1)
        output = torch.cat((z1,x1),dim=1)

        return output

class SG_Block(nn.Module):
    def __init__(self, dim=8, norm_layer=nn.LayerNorm):
        super().__init__()
        self.down = nn.Linear(768,dim)
        self.stb_x1 = SpectralTransform(dim=dim)
        self.stb_x2 = SpectralTransform(dim=dim)
        self.up = nn.Linear(dim,768)

    def forward(self,x):
        x = self.down(x)
        res_x = x
        x = self.stb_x1(x)
        # x = self.stb_x2(x)
        x = res_x + x
        x = self.up(x)
        return x

# x = torch.ones(2,320,768)
# m = SG_Block(768)
# o = m(x)
# print(o.shape)