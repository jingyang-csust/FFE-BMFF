import math
import torch
from torch import nn
import pywt

class SpectralGatingNetwork(nn.Module):
    def __init__(self, dim, h=16, w=16, SC=768):
        super().__init__()
        self.w = w
        self.h = h
        self.complex_weight = nn.Parameter(torch.randn(h, w // 2 + 1, dim, 2, dtype=torch.float32) * 0.02)

    def forward(self, x, lens_x, spatial_size=None):
        z = x[:, :lens_x, :]
        x = x[:, lens_x:, :]
        B, n, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(n))
        else:
            a, b = spatial_size
        x = x.view(B, a, b, C)
        x = x.to(torch.float32)
        print(f"x.shape: {x.shape}")

        coeffs = pywt.dwt2(x.numpy(), 'haar')
        x_wavelet, (LH, HL, HH) = coeffs

        weight = torch.view_as_complex(self.complex_weight)
        print(f"weight: {weight.shape}, x_wavelet: {x_wavelet.shape}")

        # Convert x_wavelet to a PyTorch tensor
        x_wavelet = torch.tensor(x_wavelet, dtype=torch.complex64)

        # Ensure weight and x_wavelet have compatible shapes for broadcasting
        if weight.shape != x_wavelet.shape:
            raise ValueError(f"Shape mismatch: weight.shape {weight.shape} != x_wavelet.shape {x_wavelet.shape}")

        x_wavelet = x_wavelet * weight

        # Convert x_wavelet back to a numpy array for IDWT
        x_wavelet = x_wavelet.numpy()

        x_reconstructed = pywt.idwt2((x_wavelet, (LH, HL, HH)), 'haar')
        x = torch.tensor(x_reconstructed, dtype=torch.float32)
        x = x.reshape(B, n, C)
        x = torch.cat([z, x], dim=1)

        return x

# Rename your script to avoid conflicts with the pywt module
# For example, save this script as spectral_gating.py

fft = SpectralGatingNetwork(dim=768, h=16, w=16)
x = torch.ones([2, 320, 768])
x_output = fft(x, 64)
print(x_output.shape)
