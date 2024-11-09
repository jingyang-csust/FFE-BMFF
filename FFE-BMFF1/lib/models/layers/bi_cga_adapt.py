import torch
from torch import nn
from einops.layers.torch import Rearrange
import torch.nn.init as init
from lib.utils.token_utils import patch2token,token2patch


# class SpatialAttention(nn.Module):
#     def __init__(self):
#         super(SpatialAttention, self).__init__()
#         self.sa = nn.Conv2d(2, 1, 7, padding=7 // 2, groups=1)
#
#     def forward(self, x):
#         original_shape = x.size()  # Store original shape
#
#         x_avg = torch.mean(x, dim=1, keepdim=True)
#         x_max, _ = torch.max(x, dim=1, keepdim=True)
#         x2 = torch.cat([x_avg, x_max], dim=1)
#         sattn = self.sa(x2)
#
#         # Resize sattn to match original x shape
#         target_size = (original_shape[2], original_shape[3])
#         sattn = nn.functional.interpolate(sattn, size=target_size, mode='bilinear', align_corners=False)
#
#         # Expand the dimensions to match original_shape
#         sattn = sattn.expand(original_shape[0], original_shape[1], original_shape[2], original_shape[3])
#
#         print(sattn.shape)  # Print the shape for debugging
#         return sattn


# class SpatialAttention(nn.Module):
#     def __init__(self):
#         super(SpatialAttention, self).__init__()
#         self.sa = nn.Conv2d(2, 1, 7, padding=7 // 2, groups=1)
#         self.upsample = nn.Upsample(scale_factor=None, mode='bilinear', align_corners=False)
#         self.conv = nn.Conv2d(1, 768, kernel_size=1)  # 调整通道数
#
#     def forward(self, x):
#         original_shape = x.size()  # Store original shape
#         res = x
#         x_avg = torch.mean(x, dim=1, keepdim=True)
#         x_max, _ = torch.max(x, dim=1, keepdim=True)
#         x2 = torch.cat([x_avg, x_max], dim=1)
#         sattn = self.sa(x2)
#         scale_factor_h = original_shape[2] / sattn.size(2)
#         scale_factor_w = original_shape[3] / sattn.size(3)
#         self.upsample.scale_factor = (scale_factor_h, scale_factor_w)
#         sattn = self.upsample(sattn)
#         sattn = self.conv(sattn)
#         sattn = sattn.view(original_shape)
#         return sattn + x

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 8, 7, padding=3, padding_mode='reflect' ,bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(dim // reduction),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        x_gap = self.gap(x)  # ([2, 768, 1, 1])
        cattn = self.ca(x_gap)  # ([2, 768, 1, 1])
        # cattn = nn.functional.interpolate(cattn, size=x.size()[2:], mode='bilinear', align_corners=False)
        # print(cattn.shape)  # ([2, 768, 320, 1])
        cattn = cattn.expand_as(x)
        return cattn


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 3, padding=1, groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        # print(pattn2.shape)
        return pattn2
class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_v):
        initial = x_v
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        # x_i 为高级语意信息
        result = initial + pattn2 * x_v + (1 - pattn2) * x_i
        result = self.conv(result)
        result = patch2token(result)

        return result

    # def forward(self, x_v,lens_z):
    #     z_v = x_v[:,:lens_z,:]
    #     x_v = x_v[:,lens_z:,:]
    #
    #     z_v = token2patch(z_v)
    #     x_v = token2patch(x_v)
    #
    #     x_i = token2patch(self.se(x_v))
    #     initial = x_v + x_i
    #     cattn = self.ca(initial)
    #     sattn = self.sa(initial)
    #     pattn1 = sattn + cattn
    #     pattn2 = self.sigmoid(self.pa(initial, pattn1))
    #     # x_i 为高级语意信息
    #     result = initial + pattn2 * x_v + (1 - pattn2) * x_i
    #     result = self.conv(result)
    #     result = patch2token(result)
    #
    #     z_i = token2patch(self.se(z_v))
    #     initial_z = z_v + z_i
    #     cattn_z = self.ca(initial_z)
    #     sattn_z = self.sa(initial_z)
    #     pattn1_z = sattn_z + cattn_z
    #     pattn2_z = self.sigmoid(self.pa(initial_z, pattn1_z))
    #     # z_i 为高级语意信息
    #     result_z = initial_z + pattn2_z * z_v + (1 - pattn2_z) * z_i
    #     result_z = self.conv(result_z)
    #     result_z = patch2token(result_z)
    #
    #     result_f = torch.cat((result_z,result),dim=1)
    #     return result_f




class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)



class Bi_cga_adapter(nn.Module):
    def __init__(self, dim=8, xavier_init=False):
        super().__init__()

        self.dropout = nn.Dropout(0.1)

        self.adapter_down = nn.Linear(768, dim)
        self.adapter_up = nn.Linear(dim, 768)
        self.adapter_mid = CGAFusion(dim=dim)


        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        #self.act = QuickGELU()

    def forward(self, x):
        B, N, C = x.shape
        x_down = self.adapter_down(x)
        #x_down = self.act(x_down)
        x_down = self.adapter_mid(x_down)
        #x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)
        #print("return adap x", x_up.size())
        return x_up
