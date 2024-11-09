import torch
from torch import nn

from net.attention import SEBlock


class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=3)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        return output
class FM1(nn.Module):
    def __init__(self,lch,hch):
        super(FM1, self).__init__()
        self.attH=SpatialAttention()
        self.attL=SpatialAttention()
        self.seH=SEBlock("avg",hch,2)
        self.seL=SEBlock("avg",lch,2)
        self.up=nn.ConvTranspose2d(hch, hch, kernel_size=2, stride=2)

    def forward(self, Fl,Fh):
        Fh=self.up(Fh)
        f=torch.cat([Fh,Fl],dim=1)
        attH=self.attH(f)
        attL=self.attL(f)
        Fl=self.seL(Fl)*attL
        Fh=self.seH(Fh)*attH
        out=torch.cat((Fl,Fh),dim=1)
        return out

class FM2(nn.Module):

    def __init__(self,lch,hch):
        super(FM2, self).__init__()
        self.attH=SpatialAttention()
        self.attL=SpatialAttention()
        self.seH=SEBlock("avg",hch,2)
        self.seL=SEBlock("avg",lch,2)
        self.up=nn.Upsample(scale_factor=2,
                    mode="bilinear",
                    align_corners=True)

    def forward(self, Fl,Fh):
        Fh=self.up(Fh)
        f=torch.cat([Fh,Fl],dim=1)
        attH=self.attH(f)
        attL=self.attL(f)
        Fl=self.seL(Fl)*attL
        Fh=self.seH(Fh)*attH
        out=torch.cat((Fl,Fh),dim=1)
        return out

class FM3(nn.Module):

    def __init__(self,lch,hch):
        super(FM3, self).__init__()
        self.attH=SpatialAttention()
        self.attL=SpatialAttention()
        self.seHL=SEBlock("avg",hch+lch,2)
        self.seH=SEBlock("avg",hch,2)
        self.seL=SEBlock("avg",lch,2)
        self.up=nn.ConvTranspose2d(hch, hch, kernel_size=2, stride=2)

    def forward(self, Fl,Fh):
        Fh=self.up(Fh)
        f=torch.cat([Fh,Fl],dim=1)
        f=self.seHL(f)
        attH=self.attH(f)
        attL=self.attL(f)
        Fl=self.seL(Fl)*attL
        Fh=self.seH(Fh)*attH
        out=torch.cat((Fl,Fh),dim=1)
        return out

class FM4(nn.Module):

    def __init__(self,lch,hch):
        super(FM4, self).__init__()
        self.attH=SpatialAttention()
        self.attL=SpatialAttention()
        self.seHL=SEBlock("avg",hch+lch,2)
        self.seH=SEBlock("avg",hch,2)
        self.seL=SEBlock("avg",lch,2)
        self.up=nn.Upsample(scale_factor=2,
                    mode="bilinear",
                    align_corners=True)

    def forward(self, Fl,Fh):
        Fh=self.up(Fh)
        f=torch.cat([Fh,Fl],dim=1)
        f=self.seHL(f)
        attH=self.attH(f)
        attL=self.attL(f)
        Fl=self.seL(Fl)*attL
        Fh=self.seH(Fh)*attH
        out=torch.cat((Fl,Fh),dim=1)
        return out

class UAG(nn.Module):
    def __init__(self):
        super(UAG,self).__init__()


if __name__=="__main__":
    inp1 = torch.randn([2,32,512,512])
    inp2 = torch.randn([2,64,256,256])
    model=FM2(hch=64,lch=32)
    # torchsummary.summary(model,input_size=(3,512,512),device='cpu')
    outp=model(inp1,inp2)
    print(outp.shape)