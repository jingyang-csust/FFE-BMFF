# # import torch
# # from torch import nn
# # from torch.nn.modules import activation
# # from einops import rearrange
# #
# #
# # class MultiScaleDWConv(nn.Module):
# #     def __init__(self, dim, h, w, scale=(1, 3, 5, 7)):
# #         super().__init__()
# #         self.h = h
# #         self.w = w
# #         self.scale = scale
# #         self.channels = []
# #         self.proj = nn.ModuleList()
# #         for i in range(len(scale)):
# #             if i == 0:
# #                 channels = dim - dim // len(scale) * (len(scale) - 1)
# #             else:
# #                 channels = dim // len(scale)
# #             conv = nn.Conv2d(channels, channels,
# #                              kernel_size=scale[i],
# #                              padding=scale[i] // 2,
# #                              groups=channels)
# #             self.channels.append(channels)
# #             self.proj.append(conv)
# #
# #     def forward(self, x):
# #         # x = rearrange(x,'b (h w) c -> b c h w',h=self.h, w=self.w).contiguous()
# #         x = torch.split(x, split_size_or_sections=self.channels, dim=1)
# #         out = []
# #         for i, feat in enumerate(x):
# #             out.append(self.proj[i](feat))
# #         x = torch.cat(out, dim=1)
# #         # x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
# #         return x
# # # model = MultiScaleDWConv(h=16,w=16,dim=768)
# # # x = torch.ones([1,256,768])
# # # x = model(x)
# # # print(x.shape)
# #
# # class Mlp(nn.Module):
# #     def __init__(self,
# #                  in_features,
# #                  h,
# #                  w,
# #                  hidden_features=None,
# #                  out_features=None,
# #                  drop=0, ):
# #         super().__init__()
# #         self.h = h
# #         self.w = w
# #         out_features = out_features or in_features
# #         hidden_features = hidden_features or in_features
# #         self.fc1 = nn.Linear(in_features=in_features,out_features=hidden_features)
# #         self.dwconv = MultiScaleDWConv(hidden_features,h=self.h,w=self.w)
# #         self.act = nn.GELU()
# #         self.norm = nn.BatchNorm2d(hidden_features)
# #         self.fc2 = nn.Linear(in_features=hidden_features,out_features=in_features)
# #         self.drop = nn.Dropout(drop)
# #
# #     def forward(self, x):
# #         # x = rearrange(x,'b (h w) c -> b c h w',h=self.h, w=self.w).contiguous()
# #         x = self.fc1(x)
# #
# #         x = self.dwconv(x) + x
# #         x = self.norm(self.act(x))
# #
# #         x = self.drop(x)
# #         x = self.fc2(x)
# #         x = self.drop(x)
# #
# #         # x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
# #
# #         return x
# #
# # model = Mlp(in_features=768,h=16,w=16)
# # input = torch.ones([1,256,768])
# # output = model(input)
# # print(output.shape)
# import torch
# from torch import nn
# from einops import rearrange
#
# class MultiScaleDWConv(nn.Module):
#     def __init__(self, dim, h, w, scale=(1, 3, 5, 7)):
#         super().__init__()
#         self.h = h
#         self.w = w
#         self.scale = scale
#         self.channels = []
#         self.proj = nn.ModuleList()
#         for i in range(len(scale)):
#             if i == 0:
#                 channels = dim - dim // len(scale) * (len(scale) - 1)
#             else:
#                 channels = dim // len(scale)
#             conv = nn.Conv2d(channels, channels,
#                              kernel_size=scale[i],
#                              padding=scale[i] // 2,
#                              groups=channels)
#             self.channels.append(channels)
#             self.proj.append(conv)
#
#     def forward(self, x):
#         x = torch.split(x, split_size_or_sections=self.channels, dim=1)
#         out = []
#         for i, feat in enumerate(x):
#             out.append(self.proj[i](feat))
#         x = torch.cat(out, dim=1)
#         return x
#
# class Mlp(nn.Module):
#     def __init__(self,
#                  in_features,
#                  h,
#                  w,
#                  hidden_features=None,
#                  out_features=None,
#                  drop=0, ):
#         super().__init__()
#         self.h = h
#         self.w = w
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features)
#         self.dwconv = MultiScaleDWConv(hidden_features, h=self.h, w=self.w)
#         self.act = nn.GELU()
#         self.norm = nn.BatchNorm2d(hidden_features)
#         self.fc2 = nn.Linear(in_features=hidden_features, out_features=out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x,lens_x):
#
#         batch_size, num_patches, in_features = x.shape
#         x = x.view(batch_size * num_patches, in_features)
#         x = self.fc1(x)
#         x = x.view(batch_size, num_patches, -1)
#         x = rearrange(x, 'b (h w) c -> b c h w', h=self.h, w=self.w)
#
#         x = self.dwconv(x) + x
#         x = self.norm(self.act(x))
#         x = rearrange(x, 'b c h w -> b (h w) c')
#         x = x.view(batch_size * num_patches, -1)
#
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#
#         x = x.view(batch_size, num_patches, -1)
#
#         return x
#
# model = Mlp(in_features=768, h=16, w=16)
# input = torch.ones([1, 320, 768])
# lens_x = 64
# output = model(input,lens_x)
# print(output.shape)

import torch
from torch import nn
from einops import rearrange


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

#
model = Mlp(in_features=768)
print(model)
input = torch.ones([1, 320, 768])
output = model(input)
print(output.shape)  # Expected output shape: [1, 320, 768]
