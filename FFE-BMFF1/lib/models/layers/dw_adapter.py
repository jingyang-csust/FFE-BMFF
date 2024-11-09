import torch
from torch import nn
# import timm
import math
from lib.models.layers.dwconv import MlP

'''
def forward_block(self, x):
    x = x + self.drop_path(self.attn(self.norm1(x))) + self.drop_path(self.adapter_attn(self.norm1(x))) * self.s
    x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adapter_mlp(self.norm2(x))) * self.s
    return x


def forward_block_attn(self, x):
    x = x + self.drop_path(self.attn(self.norm1(x))) + self.drop_path(self.adapter_attn(self.norm1(x))) * self.s
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x
'''


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)



class MS_Fusion(nn.Module):
    def __init__(self, h=16, w=16, dim=8, upscale_dim=1024, xavier_init=False):
        super().__init__()

        self.adapter_down = nn.Linear(768, dim)
        self.adapter_mid = MlP(in_features=dim,h=h,w=w,hidden_features=dim*2)
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

    def forward(self, x):
        # B, N, C = x.shape
        x_down = self.adapter_down(x)
        x_down = self.adapter_mid(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)

        return x_up