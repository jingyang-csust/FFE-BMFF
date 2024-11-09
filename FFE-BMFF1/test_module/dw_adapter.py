import torch
from torch import nn
# import timm
import math

class MultiScaleDWConv(nn.Module):
    def __init__(self, dim, scale=(1, 3, 5, 7)):
        super().__init__()
        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)
            conv = nn.Conv2d(channels, channels,
                             kernel_size=scale[i],
                             padding=scale[i] // 2,
                             groups=channels)
            self.channels.append(channels)
            self.proj.append(conv)

    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 drop=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.dwconv = MultiScaleDWConv(hidden_features)
        self.act = nn.ReLU()
        self.norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Linear(in_features=hidden_features, out_features=out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        batch_size, num_patches, in_features = x.shape
        x = self.fc1(x)
        x = x.permute(0, 2, 1).unsqueeze(3)

        x = self.dwconv(x) + x
        x = self.norm(self.act(x))
        x = x.squeeze(3).permute(0, 2, 1)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)


        return x



class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)



class Bi_direct_adapter(nn.Module):
    def __init__(self, dim=768 // 2, upscale_dim=1024, xavier_init=False):
        super().__init__()

        self.adapter_down = nn.Linear(768, dim)
        self.adapter_mid = Mlp(in_features=dim,hidden_features=dim)
        self.adapter_up = nn.Linear(dim, 768)

        self.adapter_upscale = nn.Linear(768, upscale_dim)
        # self.adapter_mid_upscale = nn.Linear(upscale_dim, upscale_dim)
        self.adapter_mid_upscale = Mlp(in_features=upscale_dim,hidden_features=upscale_dim)
        self.adapter_downscale = nn.Linear(upscale_dim, 768)

        #nn.init.xavier_uniform_(self.adapter_down.weight)
        # nn.init.zeros_(self.adapter_mid.bias)
        # nn.init.zeros_(self.adapter_mid.weight)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        # nn.init.zeros_(self.adapter_mid_upscale.bias)
        # nn.init.zeros_(self.adapter_mid_upscale.weight)
        nn.init.zeros_(self.adapter_upscale.weight)
        nn.init.zeros_(self.adapter_upscale.bias)
        nn.init.zeros_(self.adapter_downscale.weight)
        nn.init.zeros_(self.adapter_downscale.bias)

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

        x_upscale = self.adapter_upscale(x)
        x_upscale = self.adapter_mid_upscale(x_upscale)
        x_upscale = self.dropout(x_upscale)
        x_downscale = self.adapter_downscale(x_upscale)

        x_combined = x_up + x_downscale

        return x_combined

model = Bi_direct_adapter()
input = torch.ones([1, 320, 768])
output = model(input)
print(output.shape)