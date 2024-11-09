import torch
from torch import nn

from net.attention import SEBlock, ChannelAttention, SpatialAttention


class DED1(nn.Module):
    def __init__(self,ch):
        super(DED1,self).__init__()
        self.upConv=nn.Sequential(
            nn.Upsample(scale_factor=2,
                      mode="bilinear",
                      align_corners=True),
            nn.Conv2d(ch,ch,kernel_size=5,stride=1,padding=2),
            nn.MaxPool2d(2,2)
        )
        self.downConv=nn.Sequential(
            nn.Conv2d(2*ch,ch,1,1,0),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2,
                        mode="bilinear",
                        align_corners=True),
        )
        self.final = nn.Sequential(
            nn.Conv2d(3 * ch, ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(ch // 4, ch),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        f0=x.clone()
        f1=self.upConv(x)
        f2=torch.cat([x,f1],dim=1)
        f2=self.downConv(f2)
        f=torch.cat([f0,f1,f2],dim=1)
        out=self.final(f)
        return out

class DED2(nn.Module):
    def __init__(self,ch):
        super(DED2,self).__init__()
        self.up=nn.Upsample(scale_factor=2,
                        mode="bilinear",
                        align_corners=True)
        self.pool=nn.MaxPool2d(2,2)
        self.upconv=nn.Sequential(
            nn.Conv2d(ch+ch//2,ch,kernel_size=5,stride=1,padding=2),
            nn.MaxPool2d(2,2)
        )
        self.dowmconv=nn.Sequential(
            nn.Conv2d(ch*4,ch,kernel_size=3,stride=1,padding=1),
            nn.Upsample(scale_factor=2,
                    mode="bilinear",
                    align_corners=True)
        )
        self.conv1=nn.Conv2d(3*ch,ch,kernel_size=1,stride=1,padding=0)

    def forward(self,big,x,small):
        origin=x.clone()
        x=self.up(x)
        big=torch.cat([x,big],dim=1)
        f1=self.upconv(big)
        f1=torch.cat([origin,f1],dim=1)

        f2=self.pool(f1)
        f2=torch.cat([f2,small],dim=1)
        f2=self.dowmconv(f2)

        out=torch.cat([f1,f2],dim=1)
        out=self.conv1(out)
        return out

class DED3(nn.Module):
    def __init__(self,ch):
        super(DED3,self).__init__()
        self.up=nn.Upsample(scale_factor=2,
                        mode="bilinear",
                        align_corners=True)
        self.pool=nn.MaxPool2d(2,2)
        self.upconv=nn.Sequential(
            nn.Conv2d(ch+ch//2,ch,kernel_size=5,stride=1,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )
        self.dowmconv=nn.Sequential(
            nn.Conv2d(ch*4,ch,kernel_size=3,stride=1,padding=1),
            nn.Upsample(scale_factor=2,
                    mode="bilinear",
                    align_corners=True)
        )
        self.conv1=nn.Conv2d(3*ch,ch,kernel_size=1,stride=1,padding=0)

    def forward(self,big,x,small):
        origin=x.clone()
        x=self.up(x)
        big=torch.cat([x,big],dim=1)
        f1=self.upconv(big)
        f1=torch.cat([origin,f1],dim=1)

        f2=self.pool(f1)
        f2=torch.cat([f2,small],dim=1)
        f2=self.dowmconv(f2)

        out=torch.cat([f1,f2],dim=1)
        out=self.conv1(out)
        return out

class DED4(nn.Module):
    def __init__(self,ch):
        super(DED4,self).__init__()
        self.up=nn.Upsample(scale_factor=2,
                        mode="bilinear",
                        align_corners=True)
        self.pool=nn.MaxPool2d(2,2)
        self.upconv=nn.Sequential(
            nn.Conv2d(ch+ch//2,ch,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.dowmconv=nn.Sequential(
            nn.Conv2d(ch*4,ch,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2,
                    mode="bilinear",
                    align_corners=True)
        )
        self.conv1=nn.Conv2d(3*ch,ch,kernel_size=1,stride=1,padding=0)

    def forward(self,big,x,small):
        origin=x.clone()
        x=self.up(x)
        big=torch.cat([x,big],dim=1)
        f1=self.upconv(big)
        f1=torch.cat([origin,f1],dim=1)

        f2=self.pool(f1)
        f2=torch.cat([f2,small],dim=1)
        f2=self.dowmconv(f2)

        out=torch.cat([f1,f2],dim=1)
        out=self.conv1(out)
        return out

class DED4_noupin(nn.Module):
    def __init__(self,ch):
        super(DED4_noupin,self).__init__()
        self.up=nn.Upsample(scale_factor=2,
                        mode="bilinear",
                        align_corners=True)
        self.pool=nn.MaxPool2d(2,2)
        self.upconv=nn.Sequential(
            nn.Conv2d(ch,ch,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.dowmconv=nn.Sequential(
            nn.Conv2d(ch*4,ch,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2,
                    mode="bilinear",
                    align_corners=True)
        )
        self.conv1=nn.Conv2d(3*ch,ch,kernel_size=1,stride=1,padding=0)

    def forward(self,x,small):
        origin=x.clone()
        x=self.up(x)
        big=x
        f1=self.upconv(big)
        f1=torch.cat([origin,f1],dim=1)

        f2=self.pool(f1)
        f2=torch.cat([f2,small],dim=1)
        f2=self.dowmconv(f2)

        out=torch.cat([f1,f2],dim=1)
        out=self.conv1(out)
        return out
class DED5(nn.Module):
    def __init__(self,ch):
        super(DED5,self).__init__()
        self.up=nn.Upsample(scale_factor=2,
                        mode="bilinear",
                        align_corners=True)
        self.pool=nn.MaxPool2d(2,2)
        self.upconv=nn.Sequential(
            nn.Conv2d(ch+ch//2,ch,kernel_size=5,stride=1,padding=2),
            nn.MaxPool2d(2,2)
        )
        self.dowmconv=nn.Sequential(
            nn.Conv2d(ch*4,ch,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2,
                    mode="bilinear",
                    align_corners=True)
        )
        self.conv1=nn.Conv2d(3*ch,ch,kernel_size=1,stride=1,padding=0)

    def forward(self,big,x,small):
        origin=x.clone()
        x=self.up(x)
        big=torch.cat([x,big],dim=1)
        f1=self.upconv(big)
        f1=torch.cat([origin,f1],dim=1)

        f2=self.pool(f1)
        f2=torch.cat([f2,small],dim=1)
        f2=self.dowmconv(f2)

        out=torch.cat([f1,f2],dim=1)
        out=self.conv1(out)
        return out

class DED6(nn.Module):
    def __init__(self,ch):
        super(DED6,self).__init__()
        self.up=nn.Upsample(scale_factor=2,
                        mode="bilinear",
                        align_corners=True)
        self.pool=nn.MaxPool2d(2,2)
        self.upconv=nn.Sequential(
            nn.Conv2d(ch+ch//2,ch,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.dowmconv=nn.Sequential(
            nn.Conv2d(ch*4,ch,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2,
                    mode="bilinear",
                    align_corners=True)
        )
        self.conv3=nn.Conv2d(ch//2,ch//2,kernel_size=3,stride=1,padding=1)
        self.conv1=nn.Conv2d(3*ch,ch,kernel_size=1,stride=1,padding=0)

    def forward(self,big,x,small):
        origin=x.clone()
        x=self.up(x)
        big=self.conv3(big)
        big=torch.cat([x,big],dim=1)
        f1=self.upconv(big)
        f1=torch.cat([origin,f1],dim=1)

        f2=self.pool(f1)
        f2=torch.cat([f2,small],dim=1)
        f2=self.dowmconv(f2)

        out=torch.cat([f1,f2],dim=1)
        out=self.conv1(out)
        return out

class DED7(nn.Module):
    def __init__(self,ch):
        super(DED7,self).__init__()
        self.pool=nn.MaxPool2d(2,2)
        self.convSA=nn.Sequential(
            nn.Conv2d(ch*3,ch*3,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
        )
        self.up=nn.Upsample(scale_factor=2,
                        mode="bilinear",
                        align_corners=True)
        self.att=nn.Sequential(
            nn.Conv2d(2,1,7,stride=1,padding=3),
            nn.Sigmoid()
        )

    def forward(self,x,small):
        origin=x.clone()
        x=self.pool(x)
        x=torch.cat([x,small],dim=1)
        x=self.convSA(x)
        x=self.up(x)
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        att = torch.cat([max_result, avg_result], 1)
        att=self.att(att)
        return origin*att

class sin1(nn.Module):
    def __init__(self,ch):
        super(sin1,self).__init__()
        self.doubleconv=nn.Sequential(
            nn.Conv2d(4,ch,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )
        self.upconv=nn.Sequential(
            nn.ConvTranspose2d(ch, ch//2, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch//2,ch//2,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch//2,ch//2,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch//2,ch,kernel_size=3,stride=2,padding=1),
            nn.ReLU(inplace=True)
        )
        self.doubleconv2 = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )
        self.downconv = nn.Sequential(
            nn.Conv2d(ch,2*ch,kernel_size=3,stride=2,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*ch, 2*ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*ch, 2*ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2*ch, ch, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.finalconv=nn.Conv2d(ch,1,kernel_size=1,stride=1,padding=0)
    def forward(self,origin,coarse):
        f=torch.cat([origin,coarse],dim=1)
        f=self.doubleconv(f)
        upatt=self.upconv(f)
        f=f*upatt
        f=self.doubleconv2(f)
        downatt=self.downconv(f)
        f=f*downatt
        f=self.finalconv(f)
        return f

class sin2(nn.Module):
    def __init__(self,ch):
        super(sin2,self).__init__()
        self.upconv=nn.Sequential(
            nn.ConvTranspose2d(ch, ch, kernel_size=2, stride=2),
            nn.Conv2d(ch,ch,kernel_size=3,stride=1,padding=1,groups=ch),
            nn.ReLU(),
            nn.Conv2d(ch,ch,kernel_size=3,stride=1,padding=1),
            nn.AvgPool2d(2,2),
            nn.Tanh(),
        )
        self.downconv = nn.Sequential(
            nn.MaxPool2d(2,2),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1,groups=ch),
            nn.ReLU(),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2,
                              mode="bilinear",
                              align_corners=True),
            nn.ReLU(),
        )
    def forward(self,x):
        # up=self.upconv(x)
        # x=x*up
        down=self.downconv(x)
        x=x+down
        return x

class sin3(nn.Module):
    def __init__(self,ch):
        super(sin3,self).__init__()
        self.up=nn.ConvTranspose2d(2*ch,ch,kernel_size=2,stride=2)
        self.conv=nn.Sequential(
            nn.Conv2d(2*ch,ch,kernel_size=3,stride=1,padding=1),
            nn.ReLU()
        )
        self.pool=nn.AdaptiveMaxPool2d(1)
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=ch, out_features=2*ch, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=2*ch, out_features=ch, bias=False),
            nn.Sigmoid()
        )
    def forward(self,big,small):
        small=self.up(small)
        f=torch.cat([big,small],dim=1)
        f=self.conv(f)
        b, c, _, _ = f.shape
        att = self.pool(f).view(b, c)
        att = self.fc_layers(att).view(b, c, 1, 1)
        out=big*att
        return out

class sin4(nn.Module):
    def __init__(self,ch):
        super(sin4,self).__init__()
        self.up=nn.ConvTranspose2d(2*ch,ch,kernel_size=2,stride=2)
        self.conv=nn.Sequential(
            nn.Conv2d(2*ch,ch,kernel_size=3,stride=1,padding=1),
            nn.ReLU()
        )
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=ch, out_features=ch//2, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=ch//2, out_features=ch, bias=False),
            nn.Sigmoid()
        )
    def forward(self,big,small):
        small=self.up(small)
        f=torch.cat([big,small],dim=1)
        f=self.conv(f)
        b, c, _, _ = f.shape
        att = self.pool(f).view(b, c)
        att = self.fc_layers(att).view(b, c, 1, 1)
        out=big*att
        return out

class sin5(nn.Module):
    def __init__(self,ch):
        super(sin5,self).__init__()
        self.up=nn.ConvTranspose2d(2*ch,ch,kernel_size=2,stride=2)
        self.conv=nn.Sequential(
            nn.Conv2d(2*ch,ch,kernel_size=3,stride=1,padding=1),
            nn.ReLU()
        )
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=2*ch, out_features=ch, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=ch, out_features=ch//2, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=ch//2, out_features=ch, bias=False),
            nn.Sigmoid()
        )
    def forward(self,big,small):
        small=self.up(small)
        f=torch.cat([big,small],dim=1)
        # f=self.conv(f)
        b, c, _, _ = f.shape
        att = self.pool(f).view(b, c)
        att = self.fc_layers(att).view(b, c//2, 1, 1)
        out=big*att
        return out

class sin6(nn.Module):
    def __init__(self,ch):
        super(sin6,self).__init__()
        self.up=nn.Upsample(scale_factor=2,
                              mode="bilinear",
                              align_corners=True)
        self.conv=nn.Sequential(
            nn.Conv2d(2*ch,ch,kernel_size=3,stride=1,padding=1),
            nn.ReLU()
        )
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=3*ch, out_features=ch, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=ch, out_features=ch, bias=False),
            nn.Sigmoid()
        )
    def forward(self,big,small):
        small=self.up(small)
        f=torch.cat([big,small],dim=1)
        # f=self.conv(f)
        b, c, _, _ = f.shape
        att = self.pool(f).view(b, c)
        att = self.fc_layers(att).view(b, c//3, 1, 1)
        out=big*att
        return out

class sin7(nn.Module):
    def __init__(self,ch):
        super(sin7,self).__init__()
        self.up=nn.Upsample(scale_factor=2,
                              mode="bilinear",
                              align_corners=True)
        self.conv=nn.Sequential(
            nn.Conv2d(3*ch,ch,kernel_size=3,stride=1,padding=1),
            nn.ReLU()
        )
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=ch, out_features=ch//2, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=ch//2, out_features=ch, bias=False),
            nn.Sigmoid()
        )
    def forward(self,big,small):
        small=self.up(small)
        f=torch.cat([big,small],dim=1)
        f=self.conv(f)
        b, c, _, _ = f.shape
        att = self.pool(f).view(b, c)
        att = self.fc_layers(att).view(b, c, 1, 1)
        out=big*att
        return out

class AG(nn.Module):
    def __init__(self,ch):
        super(AG,self).__init__()
        self.conven=nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0)
        self.convde=nn.Sequential(
            nn.Conv2d(2*ch,ch,kernel_size=1,stride=1,padding=0),
            nn.Upsample(scale_factor=2,
                        mode="bilinear",
                        align_corners=True)
        )
        self.aca=nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0),
            nn.Sigmoid(),
        )

    def forward(self,big,small):
        att=self.convde(small)+self.conven(big)
        att=self.aca(att)
        return att*big



class FA(nn.Module):
    def __init__(self,ch):
        super(FA,self).__init__()
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgMLP=nn.Sequential(
            nn.Linear(ch,ch//2,bias=True),
            nn.ReLU(),
            nn.Linear(ch//2,ch)
        )
        self.maxMLP = nn.Sequential(
            nn.Linear(ch, ch // 2, bias=True),
            nn.ReLU(),
            nn.Linear(ch // 2, ch)
        )
    def forward(self,x):
        b, c, _, _ = x.shape
        avgatt=self.avgpool(x).view(b,c)
        avgatt=self.avgMLP(avgatt).view(b,c,1,1)
        maxatt = self.maxpool(x).view(b, c)
        maxatt = self.maxMLP(maxatt).view(b, c, 1, 1)
        att=avgatt+maxatt
        return x*torch.sigmoid(att)

class FA1(nn.Module):
    def __init__(self,ch):
        super(FA1,self).__init__()
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgMLP=nn.Sequential(
            nn.Linear(ch,ch//2,bias=True),
            nn.ReLU(),
            nn.Linear(ch//2,ch)
        )
        self.maxMLP = nn.Sequential(
            nn.Linear(ch, ch // 2, bias=True),
            nn.ReLU(),
            nn.Linear(ch // 2, ch)
        )
    def forward(self,x):
        b, c, _, _ = x.shape
        avgatt=self.avgpool(x).view(b,c)
        avgatt=self.avgMLP(avgatt).view(b,c,1,1)
        maxatt = self.maxpool(x).view(b, c)
        maxatt = self.maxMLP(maxatt).view(b, c, 1, 1)
        att=avgatt+maxatt
        return x+x*torch.sigmoid(att)




class DoubleAtt(nn.Module):
    def __init__(self,ch,sa="DSA"):
        super(DoubleAtt,self).__init__()
        self.CA=ChannelAttention(ch)
        if sa == ("DSA"):
            self.SA=DSA()
        elif sa == ("DSA1"):
            self.SA=DSA1()
        elif sa == ("DSA2"):
            self.SA=DSA2()
        elif sa == ("SA"):
            self.SA=SpatialAttention()
    def forward(self,x):
        short=x.clone()
        x=self.CA(x)
        x=self.SA(x)
        return short+x

class DoubleAtt1(nn.Module):
    def __init__(self,ch,sa="DSA"):
        super(DoubleAtt1,self).__init__()
        self.CA=ChannelAttention(ch)
        if sa == ("DSA"):
            self.SA=DSA()
        elif sa == ("DSA1"):
            self.SA=DSA1()
        elif sa == ("DSA2"):
            self.SA=DSA2()
        elif sa == ("SA"):
            self.SA=SpatialAttention()
    def forward(self,x):
        short=x.clone()
        x=self.SA(x)
        x = self.CA(x)
        return short+x
if __name__=="__main__":
    big = torch.randn([2,16,512,512])
    x = torch.randn([2,32,256,256])
    small = torch.randn([2,64,128,128])
    x1= torch.randn([2,1,512,512])
    x2= torch.randn([2,1,512,512])
    # model=AG(128)
    # # torchsummary.summary(model,input_size=(3,512,512),device='cpu')
    # # model.summary()
    # outp=model(x,small)
    # # print(outp.shape)
    # model=AAF1(16,4)
    # outp = model(big,small)
    # print(outp.shape)
