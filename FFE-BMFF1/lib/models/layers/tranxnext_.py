import torch
from torch import nn

from lib.models.layers.adapter_upgtaded3 import ECAAttention
from lib.utils.token_utils import patch2token, token2patch


class se_block(nn.Module):
    def __init__(self, channel, ratio=16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均池化
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),  # 全连接
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),  # 全连接
            nn.Sigmoid()
        )

    def forward(self, x):
        lens_z = 64
        x = x[:,lens_z:,:]
        z = x[:,:lens_z,:]

        x = token2patch(x)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        x = patch2token(x * y)

        z = token2patch(z)
        b, c, _, _ = z.size()
        zy = self.avg_pool(z).view(b, c)
        zy = self.fc(zy).view(b, c, 1, 1)
        z = patch2token(z * zy)

        return torch.cat((z,x),dim=1)

class ConvolutionalGLU2(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)  # 512
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.se = se_block(out_features)

    def forward(self, x):
        x = self.norm(x)
        x1= self.fc1(x)
        x1 = self.act(x1)
        x1 = self.drop(x1)  # ([2, 256, 1024])
        x1 = self.fc2(x1)
        x2 = self.se(x1)
        x = x1 * x2
        x = self.drop(x)
        return x

class MS_Fusion(nn.Module):
    def __init__(self, dim=8, xavier_init=False):
        super().__init__()

        self.adapter_down = nn.Linear(768, dim)
        self.adapter_mid = nn.Linear(dim,dim)
        self.adapter_up = nn.Linear(dim, 768)

        self.mlp = ConvolutionalGLU2(in_features=768,hidden_features=dim,out_features=768)
        self.sum = ECAAttention(kernel_size=3)

        #self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        # B, N, C = x.shape
        x_down = self.adapter_down(x)
        x_down = self.adapter_mid(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)

        x_oth = self.mlp(x)

        x_up = self.sum(x_oth + x_up)

        return x_up

# x = torch.ones(2,320,768)
# m = MS_Fusion()
# o = m(x)
# print(o.shape)