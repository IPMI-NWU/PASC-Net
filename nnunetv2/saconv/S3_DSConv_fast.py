from nnunetv2.saconv.S3_DSConv_pro import DSConv_pro
import torch.nn as nn
import torch
#xyzw方向
class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size,stride):
        super(DSConv, self).__init__()

        self.conv0 = DSConv_pro(in_channels, out_channels//4, kernel_size=kernel_size, stride=stride,morph=0,device='cuda')
        self.conv1 = DSConv_pro(in_channels, out_channels//4, kernel_size=kernel_size, stride=stride,morph=1,device='cuda')
        self.conv2 = DSConv_pro(in_channels, out_channels//4, kernel_size=kernel_size, stride=stride,morph=2,device='cuda')
        self.conv3 = DSConv_pro(in_channels, out_channels//4, kernel_size=kernel_size, stride=stride,morph=3,device='cuda')

    def forward(self, x):
        #print(self.kernel_size,self.stride,self.padding)
        x0=self.conv0(x)
        x1=self.conv1(x)
        x2=self.conv2(x)
        x3=self.conv3(x)
        return torch.cat((x0,x1,x2,x3),1)
        
##zw方向
#class DSConv(nn.Module):
#    def __init__(self, in_channels, out_channels,kernel_size,stride):
#        super(DSConv, self).__init__()
#        self.conv2 = DSConv_pro(in_channels, out_channels//2, kernel_size=kernel_size, stride=stride,morph=2,device='cuda')
#        self.conv3 = DSConv_pro(in_channels, out_channels//2, kernel_size=kernel_size, stride=stride,morph=3,device='cuda')
#
#    def forward(self, x):
#        x2=self.conv2(x)
#        x3=self.conv3(x)
#        return torch.cat((x2,x3),1)
        
#xy方向
#class DSConv(nn.Module):
#    def __init__(self, in_channels, out_channels,kernel_size,stride):
#        super(DSConv, self).__init__()
#        self.conv0 = DSConv_pro(in_channels, out_channels//2, kernel_size=kernel_size, stride=stride,morph=0,device='cuda')
#        self.conv1 = DSConv_pro(in_channels, out_channels//2, kernel_size=kernel_size, stride=stride,morph=1,device='cuda')
#
#    def forward(self, x):
#        #print(self.kernel_size,self.stride,self.padding)
#        x0=self.conv0(x)
#        x1=self.conv1(x)
#        return torch.cat((x0,x1),1)