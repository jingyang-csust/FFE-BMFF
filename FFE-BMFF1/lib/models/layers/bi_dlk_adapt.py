import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils.token_utils import patch2token,token2patch

class DLK(nn.Module):
    """
        输入：B C H W
        输出：B C H W
    """
    def __init__(self, dim):
        super().__init__()
        self.att_conv1 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim)
        self.att_conv2 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=9, groups=dim, dilation=3)
        self.linear = nn.Linear(dim * 2,dim)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.MaxPool2d(1)
        self.spatial_se = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        att1 = self.att_conv1(x)
        att2 = self.att_conv2(att1)

        att = torch.cat([att1, att2], dim=1)  # ([2, 1536, 16, 16])
        att = token2patch(self.linear(patch2token(att)))  # ([2, 768, 16, 16])

        avg_att = self.avg(att) #  ([2, 768, 16, 16])
        avg_att = avg_att.expand_as(x)
        max_att = self.max(att) # ([2, 768, 16, 16])

        att = torch.cat([avg_att, max_att], dim=1)  # ([2, 1536, 16, 16])
        att = self.spatial_se(att)  # ([2, 768, 16, 16])
        # output = att1 * att[:, 0, :, :].unsqueeze(1) + att2 * att[:, 1, :, :].unsqueeze(1)
        output = att1 * att + att2 * att
        output = output + x

        # output = output.permute(0, 2, 3, 1).reshape(B, N, C)  # 重塑形状
        return output

# x = torch.zeros([2,768,16,16])
# dim = x.shape[1]
# models = DLK(dim=dim)
# out = models(x)
# print(out.shape)

class DLKModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()
        self.spatial_gating_unit = DLK(dim)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):

        # print(x.shape)  [2, 768, 16, 16]
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.act(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        return x

class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DLKBlock(nn.Module):
    def __init__(self, dim, hx=16, wx=16,
                 hz=8,wz=8,drop_path=0.):
        super().__init__()
        self.hx = hx
        self.wx = wx
        self.hz = hz
        self.wz = wz
        self.norm_layer = nn.BatchNorm2d(dim)
        self.channels = dim
        # self.linear = nn.Linear(dim * 2, self.channels)
        self.norm_layer = nn.BatchNorm2d(dim)
        self.norm_layer1 = nn.BatchNorm2d(dim)
        self.attn = DLKModule(dim)
        self.attn1 = DLKModule(dim)
        self.mlp = Mlp(dim)
        self.mlp1 = Mlp(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        layer_scale_init_value = 1e-6
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x_v,x_i,lens_z):
        z_v = x_v[:,:lens_z,:]
        x_v = x_v[:,lens_z:,:]
        z_i = x_i[:,:lens_z,:]
        x_i = x_i[:,lens_z:,:]

        z_v = token2patch(z_v)
        x_v = token2patch(x_v)
        z_i = token2patch(z_i)
        x_i = token2patch(x_i)  # ([2, 768, 16, 16])

        # 融合rgb和rgbt的search
        # x = torch.cat((x_v,x_i),dim=1)  # ([2, 1536, 16, 16])
        # x = x.flatten(2).transpose(1, 2).contiguous()  #  ([2, 256, 1536])
        # x = self.linear(x)  # ([2, 256, 768])
        x = x_v + x_i
        x = patch2token(x)

        shortcut = x.clone()
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, self.hx, self.wx)
        x = (self.attn(x)).flatten(2).transpose(1, 2).contiguous()
        x = shortcut + self.drop_path(self.layer_scale * x)
        shortcut = x.clone()
        x = ((self.norm_layer(x.transpose(1, 2).view(B, C, self.hx, self.wx)))
             .flatten(2).transpose(1, 2).contiguous())
        x = self.mlp(x)
        x = shortcut + self.drop_path(self.layer_scale * x)

        # 融合rgb和rgbt的search
        # z = torch.cat((z_v, z_i), dim=1)  # ([2, 1536, 8, 8])
        # z = z.flatten(2).transpose(1, 2).contiguous()  # ([2, 64, 1536])
        # z = self.linear(z)  # ([2, 64, 768])
        z = z_v + z_i  # ([2, 768, 8, 8])
        z = patch2token(z)  # ([2, 64, 768])
        shortcut = z.clone()
        z = z.transpose(1, 2).view(B, C, self.hz, self.wz) # ([2, 768, 8, 8])
        z = (self.attn(z)).flatten(2).transpose(1, 2).contiguous() # ([2, 64, 768])
        z = shortcut + self.drop_path(self.layer_scale * z) # ([2, 64, 768])
        shortcut = z.clone()
        z = ((self.norm_layer(z.transpose(1, 2).view(B, C, self.hz, self.wz)))
             .flatten(2).transpose(1,2).contiguous())  # ([2, 64, 768])
        z = self.mlp(z)
        z = shortcut + self.drop_path(self.layer_scale * z)

        xo = torch.cat((z,x),dim=1)
        return xo
    # def forward(self, x_v,lens_z):
    #     z_v = x_v[:,:lens_z,:]
    #     x_v = x_v[:,lens_z:,:]
    #
    #     z_v = token2patch(z_v)
    #     x_v = token2patch(x_v) # ([2, 768, 16, 16])
    #
    #     # 融合rgb和rgbt的search
    #     # x = torch.cat((x_v,x_i),dim=1)  # ([2, 1536, 16, 16])
    #     # x = x.flatten(2).transpose(1, 2).contiguous()  #  ([2, 256, 1536])
    #     # x = self.linear(x)  # ([2, 256, 768])
    #     x = x_v
    #     # x = patch2token(x)
    #
    #     shortcut = x.clone()
    #     # B, C, H, W = x.shape
    #     # x = x.transpose(1, 2).view(B, C, self.hx, self.wx)
    #     x = (self.attn(x)).flatten(2).transpose(1, 2).contiguous()
    #     print(x.shape)
    #     print(self.layer_scale.shape)
    #     x = shortcut + self.drop_path(self.layer_scale * x)
    #     shortcut = x.clone()
    #
    #     x = self.norm_layer(token2patch(x).flatten(2).transpose(1, 2).contiguous())
    #     x = self.mlp(x)
    #     x = shortcut + self.drop_path(self.layer_scale * x)
    #
    #     # 融合rgb和rgbt的search
    #     # z = torch.cat((z_v, z_i), dim=1)  # ([2, 1536, 8, 8])
    #     # z = z.flatten(2).transpose(1, 2).contiguous()  # ([2, 64, 1536])
    #     # z = self.linear(z)  # ([2, 64, 768])
    #     z = z_v  # ([2, 768, 8, 8])
    #     # B, C, h, w = z.shape
    #     # z = patch2token(z)  # ([2, 64, 768])
    #     shortcut = z.clone()
    #     # z = z.transpose(1, 2).view(B, C, self.hz, self.wz) # ([2, 768, 8, 8])
    #     z = (self.attn1(z)).flatten(2).transpose(1, 2).contiguous() # ([2, 64, 768])
    #     z = shortcut + self.drop_path(self.layer_scale * z) # ([2, 64, 768])
    #     shortcut = z.clone()
    #     z = self.norm_layer1(token2patch(z).flatten(2).transpose(1,2).contiguous())  # ([2, 64, 768])
    #     z = self.mlp1(z)
    #     z = shortcut + self.drop_path(self.layer_scale * z)
    #
    #     xo = torch.cat((z,x),dim=1)
    #     return xo



# x = torch.zeros([2,320,768])
# xi = torch.zeros([2,320,768])
# dim = x.shape[2]
# models = DLKBlock(dim=dim)
# out = models(x,64)
# print(out.shape)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)



class Bi_DLK_adapter(nn.Module):
    def __init__(self, dim=8, xavier_init=False):
        super().__init__()

        self.dropout = nn.Dropout(0.1)

        self.adapter_down = nn.Linear(768, dim)
        self.adapter_up = nn.Linear(dim, 768)
        self.adapter_mid = DLKBlock(dim)

        #nn.init.xavier_uniform_(self.adapter_down.weight)
        # nn.init.zeros_(self.adapter_mid.bias)
        # nn.init.zeros_(self.adapter_mid.weight)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        #self.act = QuickGELU()
        self.dim = dim

    def forward(self, x,xi,lens_z):
        B, N, C = x.shape
        x_down = self.adapter_down(x)
        xi_down = self.adapter_down(xi)
        #x_down = self.act(x_down)
        x_down = self.adapter_mid(x_down,xi_down,lens_z)
        #x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)
        #print("return adap x", x_up.size())
        return x_up

