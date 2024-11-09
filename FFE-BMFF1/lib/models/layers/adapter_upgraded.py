import torch
from torch import nn

from lib.models.layers.shaf import ShuffleAttention
from lib.models.layers.sin_shaf_advan import SHA_Fusion
from lib.models.layers.transnext import ConvolutionalGLU1


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Bi_direct_adapter(nn.Module):
    def __init__(self, dim=8, xavier_init=False):
        super().__init__()

        self.adapter_down = nn.Linear(768, dim)
        self.adapter_up = nn.Linear(dim, 768)
        self.adapter_mid = nn.Linear(dim, dim)
        self.shaf = SHA_Fusion(768)
        self.norm = nn.LayerNorm(768)
        self.mlp = ConvolutionalGLU1()

        #nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_mid.bias)
        nn.init.zeros_(self.adapter_mid.weight)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        #self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        x_attn = self.shaf(x)

        x_down = self.adapter_down(x)
        x_down = self.adapter_mid(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)

        x = x_up + x_attn
        x = self.norm(self.mlp(x))
        return x
