import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from lib.utils.token_utils import patch2token


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
        self.fc2 = nn.Linear(c2, c1)

        # Initialize fc1 layer with Kaiming initialization
        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x, H, W) :
        x = self.fc1(x)
        x = self.pwconv1(x, H, W)
        x1 = self.dwconv3(x, H, W)
        x2 = self.dwconv5(x, H, W)
        x3 = self.dwconv7(x, H, W)
        return self.fc2(F.gelu(self.pwconv2(x + x1 + x2 + x3, H, W)))

class MS_Fusion(nn.Module):
    def __init__(self, dim=8, xavier_init=False):
        super().__init__()

        self.adapter_mid_x = LSA(768,dim)
        self.adapter_mid_z = LSA(768,dim)

        #nn.init.xavier_uniform_(self.adapter_down.weight)
        # nn.init.zeros_(self.adapter_mid.bias)
        # nn.init.zeros_(self.adapter_mid.weight)

        # nn.init.zeros_(self.adapter_mid_upscale.bias)
        # nn.init.zeros_(self.adapter_mid_upscale.weight)
        # nn.init.zeros_(self.adapter_up.weight)
        # nn.init.zeros_(self.adapter_up.bias)
        # nn.init.zeros_(self.adapter_down.weight)
        # nn.init.zeros_(self.adapter_down.bias)

        #self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        # B, N, C = x.shape
        z_x = x[:,:64,:]
        x_x = x[:,64:,:]

        x_down = self.adapter_mid_x(x_x,16,16)
        x_down = self.dropout(x_down)

        z_down = self.adapter_mid_z(z_x, 8, 8)
        z_down = self.dropout(z_down)

        x = torch.cat((z_down,x_down),dim=1)
        return x

# x = torch.ones([2,320,768])
# model = MS_Fusion()
# xo = model(x)
# print(xo.shape)   # torch.Size([2, 768, 16, 16])