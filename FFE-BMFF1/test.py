import torch
from torch import nn

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

# Create a sample tensor with shape [1, 1024, 768, 1]
input_tensor = torch.randn(1, 1024, 768, 1)

# Instantiate the MultiScaleDWConv module
multi_scale_dwconv = MultiScaleDWConv(dim=1024)

# Forward pass through the module
output_tensor = multi_scale_dwconv(input_tensor)

# Print the shapes before and after
print("Input shape: ", input_tensor.shape)
print("Output shape: ", output_tensor.shape)
