import torch
from torch import nn

from net.attention import SEBlock, ChannelAttentionBlock, SpatialAttentionBlock
from net.modules import crossblock, DoubleConvBlock
from testnet.DeformableConv import DeformableConv2d
from testnet.testDED import DSA, DoubleAtt, DoubleAtt1
from testnet.testrefine import DeformRM_att


class AAF(nn.Module):
    def __init__(self,ch,time):
        super(AAF,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=time,
                              mode="bilinear",
                              align_corners=True),
            nn.Conv2d(ch*time,ch,kernel_size=1,stride=1,padding=0)
        )
        self.attlayers=nn.Sequential(
            nn.Mish(),
            nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0),
            nn.Sigmoid()
        )
    def forward(self,big,small):
        short=big.clone()
        small=self.up(small)
        att=big+small
        att=self.attlayers(att)
        big=att*big
        return short+big

class AAF1(nn.Module):
    def __init__(self,ch,time):
        super(AAF1,self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(ch*time,ch,kernel_size=time,stride=time),
            # nn.Conv2d(ch*time,ch,kernel_size=1,stride=1,padding=0)
        )
        self.attlayers=nn.Sequential(
            nn.Mish(),
            nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0),
            nn.Sigmoid()
        )
    def forward(self,big,small):
        short=big.clone()
        small=self.up(small)
        att=big+small
        att=self.attlayers(att)
        big=att*big
        return short+big

class AAF2(nn.Module):
    def __init__(self,ch,time):
        super(AAF2,self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(ch*time,ch,kernel_size=time,stride=time),
            # nn.Conv2d(ch*time,ch,kernel_size=1,stride=1,padding=0)
        )
        self.forget_layers=nn.Sequential(
            nn.Mish(),
            nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0),
            nn.Sigmoid()
        )

        self.update_layers = nn.Sequential(
            nn.Conv2d(2*ch, ch, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0),
        )

    def forward(self,big,small):
        short=big.clone()
        small=self.up(small)
        forget=big+small
        forget=self.forget_layers(forget)
        C_forget=forget*big

        update=torch.cat([big,small],dim=1)
        update=self.update_layers(update)
        C_forget_update=update+C_forget
        return C_forget_update+short

class AAF3(nn.Module):
    def __init__(self,ch,time):
        super(AAF3,self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(ch*time,ch,kernel_size=time,stride=time),
            # nn.Conv2d(ch*time,ch,kernel_size=1,stride=1,padding=0)
        )
        self.forget_layers=nn.Sequential(
            nn.Mish(),
            nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0),
            nn.Sigmoid()
        )

        self.update_layers = nn.Sequential(
            nn.Conv2d(2*ch, ch, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0),
        )

    def forward(self,big,small):
        short=big.clone()
        small=self.up(small)
        update=torch.cat([big,small],dim=1)
        update=self.update_layers(update)
        return update+short

class AAF4(nn.Module):
    def __init__(self,ch,time):
        super(AAF4,self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(ch*time,ch,kernel_size=time,stride=time),
            nn.ReLU(),
            # nn.Conv2d(ch*time,ch,kernel_size=1,stride=1,padding=0)
        )
        self.forget_layers=nn.Sequential(
            nn.Mish(),
            nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0),
            nn.Sigmoid()
        )

        self.update_layers = nn.Sequential(
            nn.Conv2d(2*ch, ch, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0),
        )

    def forward(self,big,small):
        short=big.clone()
        small=self.up(small)
        update=torch.cat([big,small],dim=1)
        update=self.update_layers(update)
        return update+short

class AAF5(nn.Module):
    def __init__(self,ch,time):
        super(AAF5,self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(ch*time,ch,kernel_size=time,stride=time),
            # nn.Conv2d(ch*time,ch,kernel_size=1,stride=1,padding=0)
        )
        self.forget_layers=nn.Sequential(
            nn.Mish(),
            nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0),
            nn.Sigmoid()
        )

        self.update_layers = nn.Sequential(
            nn.Conv2d(2*ch, ch, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )

    def forward(self,big,small):
        short=big.clone()
        small=self.up(small)
        update=torch.cat([big,small],dim=1)
        update=self.update_layers(update)
        return update+short

class AAF6(nn.Module):
    def __init__(self,ch,time):
        super(AAF6,self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(ch*time,ch,kernel_size=time,stride=time),
            nn.ReLU(),
            # nn.Conv2d(ch*time,ch,kernel_size=1,stride=1,padding=0)
        )
        self.forget_layers=nn.Sequential(
            nn.Mish(),
            nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0),
            nn.Sigmoid()
        )

        self.update_layers = nn.Sequential(
            nn.Conv2d(2*ch, ch, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )

    def forward(self,big,small):
        short=big.clone()
        small=self.up(small)
        update=torch.cat([big,small],dim=1)
        update=self.update_layers(update)
        return update+short

class AAF7(nn.Module):
    def __init__(self,ch,time):
        super(AAF7,self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(ch*time,ch,kernel_size=time,stride=time),
            # nn.Conv2d(ch*time,ch,kernel_size=1,stride=1,padding=0)
        )
        self.forget_layers=nn.Sequential(
            nn.Mish(),
            nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0),
            nn.Sigmoid()
        )

        self.update_layers = nn.Sequential(
            nn.Conv2d(2*ch, ch, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0),
            DSA()
        )

    def forward(self,big,small):
        short=big.clone()
        small=self.up(small)
        update=torch.cat([big,small],dim=1)
        update=self.update_layers(update)
        return update+short

class AAF8(nn.Module):
    def __init__(self,ch,time):
        super(AAF8,self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(ch*time,ch,kernel_size=time,stride=time),
            # nn.Conv2d(ch*time,ch,kernel_size=1,stride=1,padding=0)
        )
        self.forget_layers=nn.Sequential(
            nn.Mish(),
            nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0),
            nn.Sigmoid()
        )

        self.update_layers = nn.Sequential(
            nn.Conv2d(2*ch, ch, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0),
            DoubleAtt(ch)
        )

    def forward(self,big,small):
        short=big.clone()
        small=self.up(small)
        update=torch.cat([big,small],dim=1)
        update=self.update_layers(update)
        return update+short

class AAF9(nn.Module):
    def __init__(self,ch,time):
        super(AAF9,self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(ch*time,ch,kernel_size=time,stride=time),
            # nn.Conv2d(ch*time,ch,kernel_size=1,stride=1,padding=0)
        )
        self.forget_layers=nn.Sequential(
            nn.Mish(),
            nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0),
            nn.Sigmoid()
        )

        self.update_layers = nn.Sequential(
            nn.Conv2d(2*ch, ch, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0),
            DoubleAtt1(ch)
        )

    def forward(self,big,small):
        short=big.clone()
        small=self.up(small)
        update=torch.cat([big,small],dim=1)
        update=self.update_layers(update)
        return update+short
class DeformAtt(nn.Module):
    def __init__(self,ch):
        super(DeformAtt,self).__init__()
        self.deform=DeformableConv2d(ch,ch)
        self.se=SEBlock("avg",ch,ratio=4)
        self.mask=nn.Sequential(
            nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0),
            nn.Hardsigmoid(),
        )
    def forward(self,x):
        f=self.deform(x)
        f=self.se(f)
        mask=self.mask(x)
        f=f*mask

        return f+x
class DeformAtt1(nn.Module):
    def __init__(self,ch):
        super(DeformAtt1,self).__init__()
        self.deform=DeformableConv2d(ch,ch)
        self.se=SEBlock("avg",ch,ratio=4)
        self.mask=nn.Sequential(
            nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0),
            nn.Hardsigmoid(),
        )

    def forward(self,x):
        f=self.deform(x)
        f=self.se(f)
        mask=self.mask(x)
        f=f*mask
        f = torch.relu(f)
        return f+x




# class PatchMLP(nn.Module):
#     def __init__(self,ch,patchsize=[]):
#         super(PatchMLP,self).__init__()
#         self.partion1=nn.Conv2d(ch,)
class MDE(nn.Module):
    def __init__(self,ch,norm='bn'):
        super(MDE,self).__init__()
        if norm=='gn':
            self.d5 = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=5, dilation=5),
                nn.GroupNorm(ch//4,ch),
                nn.ReLU(),
            )
            self.d3 = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=3, dilation=3),
                nn.GroupNorm(ch//4,ch),
                nn.ReLU(),
            )
            self.d1 = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(ch//4,ch),
                nn.ReLU(),
            )
            self.conv1 = nn.Sequential(
                nn.Conv2d(2 * ch,ch, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(ch//4,ch),
                nn.ReLU(),
            )
            self.conv2 = nn.Conv2d(2 * ch, ch, kernel_size=1, stride=1, padding=0)
        elif norm=='bn':
            self.d5=nn.Sequential(
                nn.Conv2d(ch,ch,kernel_size=3,stride=1,padding=5,dilation=5),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
            )
            self.d3 = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=3, dilation=3),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
            )
            self.d1 = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
            )
            self.conv1=nn.Sequential(
                nn.Conv2d(2*ch, ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
            )
            self.conv2 = nn.Conv2d(2*ch, ch, kernel_size=1, stride=1, padding=0)
    def forward(self,x):
        d5=self.d5(x)
        d3=self.d3(x)
        d1=self.d1(x)
        d35=torch.cat([d5,d3],dim=1)
        d35=self.conv1(d35)
        d135=torch.cat([d35,d1],dim=1)
        d135=self.conv2(d135)
        return d135
class MDE1(nn.Module):
    def __init__(self,ch,norm='bn'):
        super(MDE1,self).__init__()
        self.se=SEBlock("avg",2*ch,2)
        if norm=='gn':
            self.d5 = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=5, dilation=5),
                nn.GroupNorm(ch//4,ch),
                nn.ReLU(),
            )
            self.d3 = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=3, dilation=3),
                nn.GroupNorm(ch//4,ch),
                nn.ReLU(),
            )
            self.d1 = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(ch//4,ch),
                nn.ReLU(),
            )
            self.conv1 = nn.Sequential(
                nn.Conv2d(2 * ch,ch, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(ch//4,ch),
                nn.ReLU(),
            )
            self.conv2 = nn.Conv2d(2 * ch, ch, kernel_size=1, stride=1, padding=0)
        elif norm=='bn':
            self.d5=nn.Sequential(
                nn.Conv2d(ch,ch,kernel_size=3,stride=1,padding=5,dilation=5),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
            )
            self.d3 = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=3, dilation=3),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
            )
            self.d1 = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
            )
            self.conv1=nn.Sequential(
                nn.Conv2d(2*ch, ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
            )
            self.conv2 = nn.Conv2d(2*ch, ch, kernel_size=1, stride=1, padding=0)
    def forward(self,x):
        d5=self.d5(x)
        d3=self.d3(x)
        d1=self.d1(x)
        d35=torch.cat([d5,d3],dim=1)
        d35=self.se(d35)
        d35=self.conv1(d35)
        d135=torch.cat([d35,d1],dim=1)
        d135=self.conv2(d135)
        return d135

class MDE2(nn.Module):
    def __init__(self,ch,norm='bn'):
        super(MDE2,self).__init__()
        self.se1=SEBlock("avg",2*ch,2)
        self.se2=SEBlock("avg",2*ch,2)
        if norm=='gn':
            self.d5 = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=5, dilation=5),
                nn.GroupNorm(ch//4,ch),
                nn.ReLU(),
            )
            self.d3 = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=3, dilation=3),
                nn.GroupNorm(ch//4,ch),
                nn.ReLU(),
            )
            self.d1 = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(ch//4,ch),
                nn.ReLU(),
            )
            self.conv1 = nn.Sequential(
                nn.Conv2d(2 * ch,ch, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(ch//4,ch),
                nn.ReLU(),
            )
            self.conv2 = nn.Conv2d(2*ch, ch, kernel_size=3, stride=1, padding=1)
        elif norm=='bn':
            self.d5=nn.Sequential(
                nn.Conv2d(ch,ch,kernel_size=3,stride=1,padding=5,dilation=5),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
            )
            self.d3 = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=3, dilation=3),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
            )
            self.d1 = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
            )
            self.conv1=nn.Sequential(
                nn.Conv2d(2*ch, ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
            )
            self.conv2 = nn.Conv2d(2*ch, ch, kernel_size=3, stride=1, padding=1)
    def forward(self,x):
        d5=self.d5(x)
        d3=self.d3(x)
        d1=self.d1(x)
        d35=torch.cat([d5,d3],dim=1)
        d35=self.se1(d35)
        d35=self.conv1(d35)
        d135=torch.cat([d35,d1],dim=1)
        d135=self.se2(d135)
        d135=self.conv2(d135)
        return d135

class MDE3(nn.Module):
    def __init__(self,ch,norm='bn'):
        super(MDE3,self).__init__()
        self.se=SEBlock("avg",2*ch,2)
        if norm=='gn':
            self.d5 = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=5, dilation=5),
                nn.GroupNorm(ch//4,ch),
                nn.ReLU(),
            )
            self.d3 = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=3, dilation=3),
                nn.GroupNorm(ch//4,ch),
                nn.ReLU(),
            )
            self.d1 = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(ch//4,ch),
                nn.ReLU(),
            )
            self.conv1 = nn.Sequential(
                nn.Conv2d(2 * ch,ch, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(ch//4,ch),
                nn.ReLU(),
            )
            self.conv2 = nn.Conv2d(2 * ch, ch, kernel_size=1, stride=1, padding=0)
        elif norm=='bn':
            self.d5=nn.Sequential(
                nn.Conv2d(ch,ch,kernel_size=3,stride=1,padding=5,dilation=5),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
            )
            self.d3 = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=3, dilation=3),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
            )
            self.d1 = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
            )
            self.conv1=nn.Sequential(
                nn.Conv2d(2*ch, ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
            )
            self.conv2 = nn.Conv2d(2*ch, ch, kernel_size=1, stride=1, padding=0)
    def forward(self,x):
        d5=self.d5(x)
        d3=self.d3(x)
        d1=self.d1(x)
        d35=torch.cat([d5,d3],dim=1)
        d35=self.se(d35)
        d35=self.conv1(d35)
        d135=d35+d1
        # d135=torch.cat([d35,d1],dim=1)
        # d135=self.conv2(d135)
        return d135

class MDE4(nn.Module):
    def __init__(self,ch,norm='bn'):
        super(MDE4,self).__init__()
        self.se1=SEBlock("avg",2*ch,2)
        self.se2=SEBlock("avg",2*ch,2)
        if norm=='gn':
            self.d5 = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=5, dilation=5),
                nn.GroupNorm(ch//4,ch),
                nn.ReLU(),
            )
            self.d3 = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=3, dilation=3),
                nn.GroupNorm(ch//4,ch),
                nn.ReLU(),
            )
            self.d1 = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(ch//4,ch),
                nn.ReLU(),
            )
            self.conv1 = nn.Sequential(
                nn.Conv2d(2 * ch,ch, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(ch//4,ch),
                nn.ReLU(),
            )
            self.conv2 = nn.Conv2d(2*ch, ch, kernel_size=3, stride=1, padding=1)
        elif norm=='bn':
            self.d5=nn.Sequential(
                nn.Conv2d(ch,ch,kernel_size=3,stride=1,padding=5,dilation=5),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
            )
            self.d3 = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=3, dilation=3),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
            )
            self.d1 = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
            )
            self.conv1=nn.Sequential(
                nn.Conv2d(2*ch, ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
            )
            self.conv2 = nn.Conv2d(2*ch, ch, kernel_size=3, stride=1, padding=1)
    def forward(self,x):
        d5=self.d5(x)
        d3=self.d3(x)
        d1=self.d1(x)
        d35=torch.cat([d5,d3],dim=1)
        d35=self.se1(d35)
        d35=self.conv1(d35)
        d135=torch.cat([d35,d1],dim=1)
        d135=self.se2(d135)
        d135=self.conv2(d135)
        return torch.relu(d135)

class cross_conv(nn.Module):
    def __init__(self,inc,outc,norm='bn',groups=1):
        super(cross_conv,self).__init__()
        self.cross=crossblock(inc,outc)
        self.conv=DoubleConvBlock(inc,outc,norm=norm,groups=groups)
    def forward(self,x):
        left=self.cross(x)
        right=self.conv(x)
        return right+left
class cross_conv1(nn.Module):
    def __init__(self,inc,outc,norm='bn',groups=1):
        super(cross_conv1,self).__init__()
        self.cross=crossblock(inc,outc)
        self.conv=DoubleConvBlock(inc,outc,norm=norm,groups=groups)
        self.mask=DeformRM_att()
    def forward(self,x):
        left=self.cross(x)
        right=self.conv(x)
        mask=self.mask(x)
        right=right*mask
        return right+left

class SA_forconv(nn.Module):
    def __init__(self):
        super(SA_forconv,self).__init__()
        self.conv1=nn.Conv2d(4,4,kernel_size=7,stride=1,padding=3)
        self.act=nn.Hardsigmoid()
    def forward(self,x,y):
        x_max_result, _ = torch.max(x, dim=1, keepdim=True)
        x_avg_result = torch.mean(x, dim=1, keepdim=True)
        x_result = torch.cat([x_max_result, x_avg_result], 1)
        y_max_result, _ = torch.max(y, dim=1, keepdim=True)
        y_avg_result = torch.mean(y, dim=1, keepdim=True)
        y_result = torch.cat([y_max_result, y_avg_result], 1)
        result = torch.cat([x_result, y_result], 1)

        att=self.conv1(result)
        att=torch.mean(att, dim=1, keepdim=True)
        att=self.act(att)
        return att
class cross_conv2(nn.Module):
    def __init__(self,inc,outc,norm='bn',groups=1):
        super(cross_conv2,self).__init__()
        self.cross=crossblock(inc,outc)
        self.conv=DoubleConvBlock(inc,outc,norm=norm,groups=groups)
        self.mask=SA_forconv()
    def forward(self,x):
        left=self.cross(x)
        right=self.conv(x)
        mask=self.mask(left,right)
        right=right*mask
        return right+left

class cross_conv3(nn.Module):
    def __init__(self,inc,outc,norm='bn',groups=1):
        super(cross_conv3,self).__init__()
        self.cross=crossblock(inc,outc)
        self.conv=DoubleConvBlock(inc,outc,norm=norm,groups=groups)
        self.fc1=nn.Sequential(
            nn.Linear(outc,outc*2),
            nn.Linear(outc*2,outc),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(outc, outc * 2),
            nn.Linear(outc * 2, outc),
        )
    def forward(self,x):
        left=self.cross(x)
        right=self.conv(x)
        U=left+right
        U=torch.mean(U,-1)
        U=torch.mean(U,-1)
        left_att=self.fc1(U).unsqueeze(dim=1)
        right_att=self.fc2(U).unsqueeze(dim=1)
        att=torch.cat([left_att,right_att],dim=1)
        att=torch.softmax(att,dim=1)
        att_left=att[:,0,:].squeeze(dim=1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        att_right=att[:,1,:].squeeze(dim=1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        return left*att_left+right*att_right

class cross_conv4(nn.Module):
    def __init__(self,inc,outc,norm='gn',groups=1):
        super(cross_conv4,self).__init__()
        self.cross=crossblock(inc,outc)
        self.pre_conv=nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(outc // 4, outc),
            nn.ReLU(),
        )
        self.conv=nn.Sequential(
            nn.Conv2d(outc, outc*2, kernel_size=1, stride=1, padding=0,bias=False),
            nn.Conv2d(outc*2,outc*2,kernel_size=3,stride=1,padding=1,groups=outc*2,bias=False),
            nn.GroupNorm(outc*2//4,outc*2),
            SEBlock("avg",outc*2,1),
            nn.ReLU(),
            nn.Conv2d(outc*2,outc,kernel_size=1,stride=1,padding=0,bias=False)
        )
        self.mask=DeformRM_att()
    def forward(self,x):
        left=self.cross(x)
        right1=self.pre_conv(x)
        right=self.conv(right1)+right1
        mask=self.mask(x)
        right=right*mask
        return right+left

class cross_conv5(nn.Module):
    def __init__(self,inc,outc,norm='bn',groups=1):
        super(cross_conv5,self).__init__()
        self.cross=crossblock(inc,outc)
        self.conv=DoubleConvBlock(inc,outc,norm=norm,groups=groups)
        self.mask_right=DeformRM_att()
        self.mask_left=DeformRM_att()
    def forward(self,x):
        left=self.cross(x)
        right=self.conv(x)
        mask_right=self.mask_right(x)
        right=right*mask_right
        mask_left = self.mask_left(x)
        left = left * mask_left
        return right+left

class cross_conv6(nn.Module):
    def __init__(self,inc,outc,norm='bn',groups=1):
        super(cross_conv6,self).__init__()
        self.outc=outc
        self.cross=crossblock(inc,outc)
        self.conv=DoubleConvBlock(inc,outc,norm=norm,groups=groups)
        self.mask=DeformRM_att()
        self.se=SEBlock('avg',2*outc,1)
    def forward(self,x):
        left=self.cross(x)
        right=self.conv(x)
        mask=self.mask(x)
        right=right*mask
        gather=torch.cat([left,right],dim=1)
        gather=self.se(gather)
        left,right=torch.split(gather,self.outc,dim=1)
        return right+left

class cross_conv7(nn.Module):
    def __init__(self,inc,outc,norm='gn',groups=1):
        super(cross_conv7,self).__init__()
        self.cross=crossblock(inc,outc)
        self.conv=DoubleConvBlock(inc,outc,norm=norm,groups=groups)
        self.mask=DeformRM_att()
    def forward(self,x):
        left=self.cross(x)
        right=self.conv(x)
        mask=self.mask(right)
        right=right*mask
        return right+left

class cross_conv8(nn.Module):
    def __init__(self,inc,outc,norm='gn',groups=1):
        super(cross_conv8,self).__init__()
        self.cross=crossblock(inc,outc)
        self.conv=DoubleConvBlock(inc,outc,norm=norm,groups=groups)
        self.mask=DeformRM_att()
    def forward(self,x):
        left=self.cross(x)
        right=self.conv(x)
        sum=left+right
        mask=self.mask(sum)
        right=right*mask
        return right+left
class cross_conv9(nn.Module):
    def __init__(self,inc,outc,norm='gn',groups=1):
        super(cross_conv9,self).__init__()
        self.cross=crossblock(inc,outc)
        self.conv=DoubleConvBlock(inc,outc,norm=norm,groups=groups)
        self.mask_r=DeformRM_att()
        self.mask_l=DeformRM_att()
    def forward(self,x):
        left=self.cross(x)
        right=self.conv(x)
        sum=left+right
        mask_r=self.mask_r(sum)
        right=right*mask_r
        mask_l = self.mask_l(sum)
        left = left * mask_l
        return right+left

class cross_conv10(nn.Module):
    def __init__(self,inc,outc,norm='bn',groups=1):
        super(cross_conv10,self).__init__()
        self.cross=crossblock(inc,outc)
        self.conv=nn.Sequential(
            DoubleConvBlock(inc,outc,norm=norm,groups=groups),
            nn.Dropout(p=0.5),
        )
        self.mask=DeformRM_att()
    def forward(self,x):
        left=self.cross(x)
        right=self.conv(x)
        mask=self.mask(x)
        right=right*mask
        return right+left

class DSA_forconv(nn.Module):
    def __init__(self):
        super(DSA_forconv,self).__init__()
        self.conv1=nn.Conv2d(2,1,kernel_size=3,stride=1,padding=1)
        self.conv2=nn.Conv2d(2,1,kernel_size=3,stride=1,padding=2,dilation=2)
        self.conv3=nn.Conv2d(2,1,kernel_size=3,stride=1,padding=3,dilation=3)
        self.act=nn.Sigmoid()
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
class cross_conv11(nn.Module):
    def __init__(self,inc,outc,norm='gn',groups=1):
        super(cross_conv11,self).__init__()
        self.cross=crossblock(inc,outc)
        self.conv=DoubleConvBlock(inc,outc,norm=norm,groups=groups)
        self.mask_r=DSA_forconv()
        self.mask_l=DSA_forconv()
    def forward(self,x):
        left=self.cross(x)
        right=self.conv(x)
        sum=left+right
        mask_r=self.mask_r(sum)
        mask_l = self.mask_l(sum)
        mask=torch.cat([mask_l,mask_r],dim=1)
        mask=torch.softmax(mask,dim=1)
        mask_l,mask_r=torch.split(mask,1,dim=1)
        right=right*mask_r
        left = left * mask_l
        return right+left

class cross_conv12(nn.Module):
    def __init__(self,inc,outc,norm='gn',groups=1):
        self.outc=outc
        super(cross_conv12,self).__init__()
        self.cross = crossblock(inc, outc)
        self.conv = DoubleConvBlock(2*outc, outc, norm=norm, groups=groups)
        self.proj=nn.Conv2d(2*outc,outc,1,bias=False)
    def forward(self,x):
        left = self.cross(x)
        _,right1,right2=torch.split(x,self.outc,dim=1)
        right=torch.cat([right1,right2],dim=1)
        right = self.conv(right)
        f=torch.cat([left,right],dim=1)
        f=self.proj(f)
        return f

class cross_conv13(nn.Module):
    def __init__(self,inc,outc,norm='gn',groups=1):
        self.outc=outc
        super(cross_conv13,self).__init__()
        self.cross = crossblock(inc, outc)
        self.conv = DoubleConvBlock(2*outc, outc, norm=norm, groups=groups)
        self.proj=nn.Conv2d(2*outc,outc,1,bias=False)
        self.mask=DSA_forconv()
    def forward(self,x):
        left = self.cross(x)
        mask=self.mask(left)
        _,right1,right2=torch.split(x,self.outc,dim=1)
        right=torch.cat([right1,right2],dim=1)
        right = self.conv(right)
        right=mask*right
        f=torch.cat([left,right],dim=1)
        f=self.proj(f)
        return f

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn

class MyLSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv_spatial = DeformableConv2d(dim,dim)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn
class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = MyLSKblock(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x

class LKBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.,drop_path=0., act_layer=nn.GELU, norm_cfg=None):
        super().__init__()

        # self.norm1 = nn.GroupNorm(dim//8,dim)
        # self.norm2 = nn.GroupNorm(dim//8,dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x +self.mlp(self.norm2(x))
        return x
class enFuse1(nn.Module):
    def __init__(self,ch):
        super(enFuse1,self).__init__()
        self.ca=ChannelAttentionBlock(ch)
        self.sa=SpatialAttentionBlock(ch)
    def forward(self,cross,conv):
        cross=self.ca(cross)
        conv=self.sa(conv)
        out=cross+conv
        return out
class enFuse2(nn.Module):
    def __init__(self,ch):
        super(enFuse2,self).__init__()
        self.ca=ChannelAttentionBlock(ch)
        self.sa=SpatialAttentionBlock(ch)
    def forward(self,cross,conv):
        cross=self.sa(cross)
        conv=self.ca(conv)
        out=cross+conv
        return out
class enFuse3(nn.Module):
    def __init__(self,ch):
        super(enFuse3,self).__init__()
        self.ca=ChannelAttentionBlock(ch)
        self.sa=SpatialAttentionBlock(ch)
    def forward(self,cross,conv):
        cross=self.ca(cross)
        conv=self.ca(conv)
        out=cross+conv
        return out
class enFuse4(nn.Module):
    def __init__(self, features, M=2,  r=2,  L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(enFuse4, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
                                nn.GroupNorm(d//4,d),
                                nn.ReLU(inplace=True))
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, features, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, x_no_orient):

        batch_size = x.shape[0]

        # feats = [conv(x) for conv in self.convs]
        feats = torch.cat([x, x_no_orient], dim=1)
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])

        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
        attention_vectors = 2*self.softmax(attention_vectors)

        feats_V = torch.sum(feats * attention_vectors, dim=1)

        return feats_V

class enFuse5(nn.Module):
    def __init__(self,ch):
        super(enFuse5,self).__init__()
        self.conv=LKBlock(ch)
    def forward(self,cross,conv):
        f=cross+conv
        f=self.conv(f)
        return f

class enFuse6(nn.Module):
    def __init__(self,ch):
        super(enFuse6,self).__init__()
    def forward(self,cross,conv):
        f=cross+conv
        return f
if __name__=="__main__":
    big = torch.randn([2,16,512,512])
    # x = torch.randn([2,32,256,256])
    x = torch.randn([2,128,64,64])
    small = torch.randn([2,64,128,128])
    x1= torch.randn([2,1,512,512])
    x2= torch.randn([2,1,512,512])
    # model=AG(128)
    # # torchsummary.summary(model,input_size=(3,512,512),device='cpu')
    # # model.summary()
    # outp=model(x,small)
    # # print(outp.shape)
    model=enFuse1(128)
    outp = model(x,x)
    print(outp.shape)