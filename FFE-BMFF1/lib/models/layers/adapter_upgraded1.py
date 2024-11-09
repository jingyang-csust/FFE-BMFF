import torch
from torch import nn

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


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


class Bi_Adapter_Lsa(nn.Module):
    def __init__(self, c1, c2=8):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.pwconv1 = DWC(c2,1)
        self.dwconv3 = DWC(c2, 3)
        self.dwconv5 = DWC(c2, 5)
        self.dwconv7 = DWC(c2, 7)
        self.pwconv2 = CB11(c2)
        self.fc2 = nn.Linear(c2, c1)

        # Initialize fc1 layer with Kaiming initialization
        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x) :
        H = 16
        W = 16
        x = self.fc1(x)
        z = x[:,:64,:]
        x = x[:,64:,:]

        x = self.pwconv1(x, H, W)
        x1 = self.dwconv3(x, H, W)
        x2 = self.dwconv5(x, H, W)
        x3 = self.dwconv7(x, H, W)

        x1 = self.pwconv2(x1,H,W)
        x2 = self.pwconv2(x2,H,W)
        x3 = self.pwconv2(x3,H,W)
        x = self.fc2(F.gelu(x + x1 + x2 + x3))

        z = self.fc2(z)
        x = torch.cat((z,x),dim=1)

        return x

# class QuickGELU(nn.Module):
#     def forward(self, x: torch.Tensor):
#         return x * torch.sigmoid(1.702 * x)
#
#
#
# class Bi_direct_adapter(nn.Module):
#     def __init__(self, dim=8, xavier_init=False):
#         super().__init__()
#
#         self.adapter_down = nn.Linear(768, dim)
#         self.adapter_up = nn.Linear(dim, 768)
#         self.adapter_mid = LSA(dim, dim)
#
#         #nn.init.xavier_uniform_(self.adapter_down.weight)
#         nn.init.zeros_(self.adapter_down.weight)
#         nn.init.zeros_(self.adapter_down.bias)
#         nn.init.zeros_(self.adapter_up.weight)
#         nn.init.zeros_(self.adapter_up.bias)
#
#         #self.act = QuickGELU()
#         self.dropout = nn.Dropout(0.1)
#         self.dim = dim
#
#     def forward(self, x):
#         B, N, C = x.shape
#         x_down = self.adapter_down(x)
#         x_down = self.adapter_mid(x_down,16,16)
#         x_down = self.dropout(x_down)
#         x_up = self.adapter_up(x_down)
#         return x_up

# x = torch.ones(2,320,768)
# m = Bi_Adapter_Lsa(768)
# o = m(x)
# print(o.shape)