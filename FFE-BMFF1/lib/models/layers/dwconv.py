import torch
from torch import nn
from einops import rearrange


class MultiScaleDWConv(nn.Module):
    def __init__(self, dim, h, w, scale=(1, 3, 5, 7)):
        super().__init__()
        self.h = h
        self.w = w
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


class MlP(nn.Module):
    def __init__(self,
                 in_features,
                 h,
                 w,
                 hidden_features=None,
                 out_features=None,
                 drop=0):
        super().__init__()
        self.h = h
        self.w = w
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.dwconv = MultiScaleDWConv(hidden_features, h=self.h, w=self.w)
        self.act = nn.GELU()
        self.norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Linear(in_features=hidden_features, out_features=out_features)
        self.drop = nn.Dropout(drop)
        self.linear = nn.Linear(out_features,out_features)
    def forward(self, x):
        batch_size, num_patches, in_features = x.shape
        lens_x = 64
        # 分割模版和搜索区域
        template = x[:, :lens_x]
        search = x[:, lens_x:]

        # 处理搜索区域
        search = search.reshape(batch_size * (num_patches - lens_x), in_features)
        search = self.fc1(search)
        search = search.view(batch_size, num_patches - lens_x, -1)
        search = rearrange(search, 'b (h w) c -> b c h w', h=self.h, w=self.w)

        search = self.dwconv(search) + search
        search = self.norm(self.act(search))
        search = rearrange(search, 'b c h w -> b (h w) c')
        search = search.view(batch_size * (num_patches - lens_x), -1)

        search = self.drop(search)
        search = self.fc2(search)
        search = self.drop(search)

        search = search.view(batch_size, num_patches - lens_x, -1)

        # search = rearrange(search, 'b (h w) c -> b c h w', h=self.h, w=self.w)
        # search = self.act(self.conv1x1(search))
        # search = rearrange(search, 'b c h w -> b (h w) c')
        # 合并模版和处理后的搜索区域
        x = torch.cat([template, search], dim=1)
        x = self.linear(x)
        return x
# x = torch.ones([32,320,768])
# m = MlP(x,16,16)
# o = m(x,64)
# print(x.shape)