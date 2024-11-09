import torch
from torch import nn
from torch.nn import init
from lib.utils.token_utils import token2patch,patch2token
class CBAMLayer(nn.Module):
    """
        输入：B C H W
        输出：B C H W
    """
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x
# x = torch.randn([2, 768, 32, 32])
# net = CBAMLayer(768)
# y = net.forward(x)
# print(y.shape)
class GAM_Attention(nn.Module):
    """
        输入：B C H W
        输出：B C H W
    """
    def __init__(self, in_channels, out_channels, rate=4):
        super(GAM_Attention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)

        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att

        return out
# x = torch.ones([2,768,16,16])
# m = GAM_Attention(768,768)
# o = m(x)
# print(o.shape)
class SE(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

        # Initialize linear layers with Kaiming initialization
        for m in self.fc:
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # B, _, C = x.shape
        # x = x.transpose(1, 2).view(B, C, H, W)
        # print(x.shape) #  ([2, 768, 16, 16])
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # print(y.shape) ([2, 768, 1, 1])
        return (x * y.expand_as(x)).flatten(2).transpose(1, 2).contiguous()

class DFF(nn.Module):
    """
    输入：B C H W
    输出：B C H W
    """
    def __init__(self, dim):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cbam = CBAMLayer(dim)

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, bias=True)
        self.act = nn.ReLU()
    def forward(self, x):

        att = self.cbam(x)
        output = x * att
        # print(output.shape)  # ([2, 768, 16, 16])

        att_up = self.conv1(x) + self.conv2(x)
        att_up = self.act(att_up)
        output = output * att_up       # ([2, 768, 16, 16])

        return output
