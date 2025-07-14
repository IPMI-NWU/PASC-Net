#三选一
#from nnunetv2.saconv.S3_DSConv_my import DSConv
#from nnunetv2.saconv.S3_DSConv_slow import DSConv
from nnunetv2.saconv.S3_DSConv_fast import DSConv


import torch
import numpy as np
import os
import time
import torch.nn as nn
class SAConv2d(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,padding_mode='zeros',bias=False):
        super(SAConv2d, self).__init__()
        #print(kernel_size,stride,padding)
        #exit()
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        if isinstance(kernel_size,list):
            kernel_size=kernel_size[0]
        if isinstance(stride,list):
            stride=stride[0]
        if isinstance(padding,list):
            padding=padding[0]
        if kernel_size==1:
            self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)
        else:
            #self.conv = DSConv(in_channels, out_channels//2, kernel_size=kernel_size*3, stride=stride,padding=padding+3,dilation=dilation,groups=groups,padding_mode=padding_mode,bias=bias)
            self.conv = DSConv(in_channels, out_channels, kernel_size=kernel_size*3, stride=stride)

    def forward(self, x):
        #print(self.kernel_size,self.stride,self.padding)
        x=self.conv(x)
        return x
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    time_start = time.time()
    A = np.random.rand(1, 3, 512, 512)
    A = A.astype(dtype=np.float32)
    A = torch.from_numpy(A)

    conv0 = SAConv(
        in_channels=3,
        out_channels=10,
        kernel_size=9,
        stride=2,
        padding=4,
        dilation=1,
        groups=1,
        padding_mode='zeros',
        bias=True,
        )
    if torch.cuda.is_available():
        A = A.to(device)
        conv0 = conv0.to(device)
    out,td = conv0(A)
    print(out.shape)
    end_time=time.time()
    diff_time=end_time-time_start
    print(diff_time)
    print("dsc占比：",td/diff_time*100,"%")