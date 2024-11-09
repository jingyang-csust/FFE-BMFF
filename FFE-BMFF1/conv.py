import torch
from torch import nn

conv=nn.Conv2d(in_channels=320,out_channels=320,kernel_size=5,stride=1,padding=0)
x = torch.ones(2,320,768,1)
o = conv(x)
print(o.shape)