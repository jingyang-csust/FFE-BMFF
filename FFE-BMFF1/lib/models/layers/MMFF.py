import torch
import torch.nn as nn
from lib.utils.token_utils import token2patch,patch2token

class CFN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        #
        self.conv55 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, stride=1, padding=2,
                                groups=in_channels)
        self.bn55 = nn.BatchNorm2d(in_channels, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)

        self.conv33 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1,
                                groups=in_channels)
        self.bn33 = nn.BatchNorm2d(in_channels, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)

        self.conv11 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0,
                                groups=in_channels)
        self.bn11 = nn.BatchNorm2d(in_channels, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)

        #
        self.conv_up = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=1, stride=1,
                                 padding=0)
        self.bn_up = nn.BatchNorm2d(in_channels * 2, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)
        self.act = nn.GELU()

        self.conv_down = nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=1, stride=1,
                                   padding=0)
        self.bn_down = nn.BatchNorm2d(in_channels, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)

        # down
        self.adjust = nn.Conv2d(in_channels, out_channels, 1)

        # norm all
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # B, N, _C = x.shape
        # x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        # print(f"x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous():{x.shape}")
        residual = self.residual(x)

        #  + skip-connection
        x = x + self.bn11(self.conv11(x)) + self.bn33(self.conv33(x)) + self.bn55(self.conv55(x))

        #  + skip-connection
        x = x + self.bn_down(self.conv_down(self.act(self.bn_up(self.conv_up(x)))))

        x = self.adjust(x)

        out = self.norm(residual + x)
        return out

class MM_FF(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.cfn1 = CFN(dim,dim)
        self.cfn2 = CFN(dim,dim)

    def forward(self,x_v,x_i,lens_z):
        z_v = x_v[:, :lens_z, :]
        x_v = x_v[:, lens_z:, :]
        z_i = x_i[:, :lens_z, :]
        x_i = x_i[:, lens_z:, :]
        z_v = token2patch(z_v)
        x_v = token2patch(x_v)
        z_i = token2patch(z_i)
        x_i = token2patch(x_i)

        merge_x = x_v + x_i
        # print(merge_x.shape)  # torch.Size([2, 768, 16, 16])
        merge_x = self.cfn1(merge_x)

        merge_z = z_v + z_i
        merge_z = self.cfn2(merge_z)

        merge_x = patch2token(merge_x)
        merge_z = patch2token(merge_z)
        x = torch.cat((merge_z,merge_x),dim=1)
        return x

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)



class MS_Fusion(nn.Module):
    def __init__(self, h=16, w=16, dim=8, upscale_dim=1024, xavier_init=False):
        super().__init__()

        self.adapter_down = nn.Linear(768, dim)
        self.adapter_mid = MM_FF(dim)
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
# x = torch.ones([2,320,768])
# model = MS_Fusion(768)
# xo = model(x,x,64)
# print(xo.shape)
