from test_module.FFT import SpectralGatingNetwork
from test_module.adapter import Bi_direct_adapter
from test_module.dwconv import MultiScaleDWConv
import torch


# fft = SpectralGatingNetwork(dim=768,h=16,w=16)
# ba = Bi_direct_adapter()
dwconv = MultiScaleDWConv(dim=768)
print(dwconv)
x = torch.ones([1, 320, 768])
xi = torch.ones([1, 320, 768])

# x_output,xi_output= fft(x,xi,64)
# x_output = ba(x_output)
# xi_output = ba(xi_output)
# x_output = dwconv(x)

print(x.shape)
# print(xi.shape)

