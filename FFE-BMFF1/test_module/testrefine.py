import torch
from torch import nn

from testnet.DSConv import DSConv
from testnet.DeformableConv import DeformableConv2d


class DeformRM_att(nn.Module):
    def __init__(self):
        super(DeformRM_att,self).__init__()
        self.conv1=nn.Conv2d(2,1,kernel_size=3,stride=1,padding=1)
        self.conv2=nn.Conv2d(2,1,kernel_size=3,stride=1,padding=3,dilation=3)
        self.conv3=nn.Conv2d(2,1,kernel_size=3,stride=1,padding=5,dilation=5)
        self.act=nn.Hardsigmoid()
    def forward(self,x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        d1=self.conv1(result)
        d2=self.conv2(result)
        d3=self.conv3(result)
        att=d1+d2+d3
        att=self.act(att)
        return att

class DeformRM(nn.Module):
    def __init__(self,ch):
        super(DeformRM,self).__init__()
        self.deform=DeformableConv2d(ch,ch)
        self.att=DeformRM_att()
    def forward(self,x):
        f=self.deform(x)
        mask=self.att(x)
        refine=f*mask
        return refine+x

class DeformRM1(nn.Module):
    def __init__(self,ch):
        device = torch.device("cuda")
        super(DeformRM1,self).__init__()
        self.x = DSConv(
            ch,
            ch,
            9,
            1,
            0,
            True,
            device=device,
        )
        self.y = DSConv(
            ch,
            ch,
            9,
            1,
            1,
            True,
            device=device,
        )
        self.attx=DeformRM_att()
        self.atty=DeformRM_att()
    def forward(self,x):
        typex=self.x(x)
        maskx=self.attx(x)
        typey = self.y(x)
        masky = self.atty(x)
        refine=typex*maskx+typey*masky
        return refine+x
