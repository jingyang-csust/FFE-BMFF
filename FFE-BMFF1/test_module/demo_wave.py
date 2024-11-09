import torch
from torch import nn
from wave_p import DWT_2D, IDWT_2D
from einops import rearrange

class Model(nn.Module):
    def __init__(self, dim, wavelet='haar'):
        super().__init__()
        self.dwt = DWT_2D(wavelet)
        self.i_dwt = IDWT_2D(wavelet)


    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = x.to(torch.float32)  # Ensure input tensor is float32

        x_dwt = self.dwt(x)
        x_i_dwt = self.i_dwt(x_dwt)
        x_i_dwt = x_i_dwt.view(B, -1, x_i_dwt.size(-2) * x_i_dwt.size(-1)).transpose(1, 2)


        return x_i_dwt

# Example usage
x = torch.ones([1, 256, 768])
dim = x.shape[2]
model = Model(dim=dim)
x = model(x, 16, 16)
print(x.shape)
