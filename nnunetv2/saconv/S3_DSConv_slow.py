from nnunetv2.saconv.S3_DSConv_old import DSConv_old
import torch.nn as nn
import torch
class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size,stride):
        super(DSConv, self).__init__()

        self.conv0 = DSConv_old(in_channels, out_channels//2, kernel_size=kernel_size, stride=stride,morph=0,device=torch.device("cuda"))
        self.conv1 = DSConv_old(in_channels, out_channels//2, kernel_size=kernel_size, stride=stride,morph=1,device=torch.device("cuda"))

    def forward(self, x):
        #print(self.kernel_size,self.stride,self.padding)
        x0=self.conv0(x)
        x1=self.conv1(x)
        return torch.cat((x0,x1),1)