import os
import torch
import numpy as np
from torch import nn
import warnings
import torch
import pickle
import torch.nn.functional as F

class DSConv(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride,padding, dilation,groups,padding_mode, bias,
                 if_offset=True):
        super(DSConv, self).__init__()
        self.kernel_size = kernel_size
        self.bn = nn.BatchNorm2d(2 * self.kernel_size)
        self.dilation=dilation
        self.padding=padding
        self.padding_mode=padding_mode
        self.dsc_conv_x = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(kernel_size, 1),
            stride= (self.kernel_size, 1),
            padding=0,
            bias=bias,
            groups=groups,
        )
        self.dsc_conv_y = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(1, kernel_size),
            stride=(1, kernel_size),
            padding=0,
            groups=groups,
        )

        self.offset_conv=nn.Conv2d(in_ch, 2 * self.kernel_size, self.kernel_size, padding=padding,stride=stride,dilation=dilation)#偏移shape与坐标shape一致
        self.stride=stride
        self.if_offset = if_offset
        self.device = torch.device("cuda")

    def forward(self, f):
        offset = self.offset_conv(f)
        offset = self.bn(offset)
        offset = torch.tanh(offset)
        if self.padding_mode != 'zeros':
            f = F.pad(f, (self.padding, self.padding, self.padding, self.padding),
                      mode=self.padding_mode)
        else:
            f = F.pad(f, (self.padding, self.padding, self.padding, self.padding),
                      mode='constant', value=0)
        input_shape = f.shape

        dsc = DSC(input_shape, self.kernel_size, self.stride, self.padding, self.dilation,
                  self.device)
        deformed_featurez, deformed_featurew = dsc.deform_conv(f, offset, self.if_offset)
        x2 = self.dsc_conv_x(deformed_featurez)
        x3 = self.dsc_conv_y(deformed_featurew)
        return torch.cat((x2, x3),1)


class DSC(object):

    def __init__(self, input_shape, kernel_size, stride,padding, dilation, device):
        self.num_points = kernel_size
        self.width = input_shape[2]#行/高
        self.height = input_shape[3]#列/宽
        self.channels=input_shape[1]
        self.device = device
        self.stride=stride
        self.padding=padding
        self.dilation=dilation

        self.h_k = int((self.height - ((self.num_points-1)*self.dilation+1) + 1) / stride)
        self.w_k = int((self.width - ((self.num_points-1)*self.dilation+1) + 1) / stride)
        # define feature map shape
        self.num_batch = input_shape[0]
        self.num_channels = input_shape[1]


    def _coordinate_map_3D(self, offset, if_offset):
        z_offset,w_offset = torch.split(offset, self.num_points, dim=1)
        d=int(((self.num_points-1)*self.dilation+1)/2)
        y_center = torch.arange(0+d, self.width-d,self.stride).repeat([self.h_k])
        y_center = y_center.reshape(self.h_k, self.w_k)
        y_center = y_center.permute(1, 0)
        y_center = y_center.reshape([-1, self.w_k, self.h_k])
        y_center = y_center.repeat([self.num_points, 1, 1]).float()
        y_center = y_center.unsqueeze(0)

        x_center = torch.arange(0+d, self.height-d,self.stride).repeat([self.w_k])
        x_center = x_center.reshape(self.w_k, self.h_k)
        x_center = x_center.permute(0, 1)
        x_center = x_center.reshape([-1, self.w_k, self.h_k])
        x_center = x_center.repeat([self.num_points, 1, 1]).float()
        x_center = x_center.unsqueeze(0)

        y_2 = torch.linspace(
            int(self.num_points // 2)*self.dilation,
            -int(self.num_points // 2)*self.dilation,
            int(self.num_points),
        )
        x_2 = torch.linspace(
            -int(self.num_points // 2)*self.dilation,
            int(self.num_points // 2)*self.dilation,
            int(self.num_points),
        )
        z = torch.linspace(0, 0, 1)

        y_2, _ = torch.meshgrid(y_2, z)
        y_spread_2 = y_2.reshape(-1, 1)
        x_2, _ = torch.meshgrid(x_2, z)
        x_spread_2 = x_2.reshape(-1, 1)

        y_grid_2 = y_spread_2.repeat([1, self.w_k * self.h_k])
        y_grid_2 = y_grid_2.reshape([self.num_points, self.w_k, self.h_k])
        y_grid_2 = y_grid_2.unsqueeze(0)  # [B*K*K, W,H]

        x_grid_2 = x_spread_2.repeat([1, self.w_k * self.h_k])
        x_grid_2 = x_grid_2.reshape([self.num_points, self.w_k, self.h_k])
        x_grid_2 = x_grid_2.unsqueeze(0)  # [B*K*K, W,H]

        y_new_2 = y_center + y_grid_2
        x_new_2 = x_center + x_grid_2

        y_new_2 = y_new_2.repeat(self.num_batch, 1, 1, 1).to(self.device)
        x_new_2 = x_new_2.repeat(self.num_batch, 1, 1, 1).to(self.device)
        
        y_offset_new_2 = z_offset.detach().clone()
        if if_offset:
            y_offset_new_2 = y_offset_new_2.permute(1, 0, 2, 3)
            z_offset = z_offset.permute(1, 0, 2, 3)
            center = int(self.num_points // 2)
            head=0
            tail=self.num_points-1
            y_offset_new_2[head] = 0
            y_offset_new_2[tail] = 0

            for index in range(1, center):
                y_offset_new_2[head + index] = (
                        y_offset_new_2[head + index - 1] + z_offset[head + index] )
                y_offset_new_2[tail - index] = (
                        y_offset_new_2[tail - index + 1] + z_offset[tail - index] )

            y_offset_new_2[4]=(y_offset_new_2[3]+y_offset_new_2[5])/2
            y_offset_new_2 = y_offset_new_2.permute(1, 0, 2, 3).to(self.device)
            y_new_2 = y_new_2.add(y_offset_new_2)

            x_new_2 = x_new_2.add(y_offset_new_2)

        y_new_2 = y_new_2.reshape(
            [self.num_batch, self.num_points, 1, self.w_k, self.h_k])
        y_new_2 = y_new_2.permute(0, 3, 1, 4, 2)#num_batch,w_k,num_points,h_k,1
        y_new_2 = y_new_2.reshape([
            self.num_batch, self.num_points * self.w_k, 1 * self.h_k
        ])
        
        x_new_2 = x_new_2.reshape(
            [self.num_batch, self.num_points, 1, self.w_k, self.h_k])
        x_new_2 = x_new_2.permute(0, 3, 1, 4, 2)
        x_new_2 = x_new_2.reshape([
            self.num_batch, self.num_points * self.w_k, 1 * self.h_k
        ])

        """
       Initialize the kernel and flatten the kernel
           y: -num_points//2 ~ num_points//2
           x: -num_points//2 ~ num_points//2 (Determined by the kernel size)
           !!! The related PPT will be submitted later, and the PPT will contain the whole changes of each step
       """
        y_3 = torch.linspace(
            -int(self.num_points // 2)*self.dilation,
            int(self.num_points // 2)*self.dilation,
            int(self.num_points),
        )
        x_3 = torch.linspace(
            -int(self.num_points // 2)*self.dilation,
            int(self.num_points // 2)*self.dilation,
            int(self.num_points),
        )
        z = torch.linspace(0, 0, 1)

        y_3, _ = torch.meshgrid(y_3, z)
        y_spread_3 = y_3.reshape(-1, 1)
        x_3, _ = torch.meshgrid(x_3, z)
        x_spread_3 = x_3.reshape(-1, 1)

        y_grid_3 = y_spread_3.repeat([1, self.w_k * self.h_k])
        y_grid_3 = y_grid_3.reshape([self.num_points, self.w_k, self.h_k])
        y_grid_3 = y_grid_3.unsqueeze(0)  # [B*K*K, W,H]

        x_grid_3 = x_spread_3.repeat([1, self.w_k * self.h_k])
        x_grid_3 = x_grid_3.reshape([self.num_points, self.w_k, self.h_k])
        x_grid_3 = x_grid_3.unsqueeze(0)  # [B*K*K, W,H]

        y_new_3 = y_center + y_grid_3
        x_new_3 = x_center + x_grid_3

        y_new_3 = y_new_3.repeat(self.num_batch, 1, 1, 1).to(self.device)
        x_new_3 = x_new_3.repeat(self.num_batch, 1, 1, 1).to(self.device)
        y_offset_new_3 = z_offset.detach().clone()

        if if_offset:
            w_offset = w_offset.permute(1, 0, 2, 3)

            center = int(self.num_points // 2)
            head=0
            tail=self.num_points-1

            y_offset_new_3[head] = 0
            y_offset_new_3[tail] = 0

            for index in range(1, center):
                y_offset_new_3[head + index] = (
                            y_offset_new_3[head + index - 1] + w_offset[head + index] )
                y_offset_new_3[tail - index] = (
                            y_offset_new_3[tail - index + 1] + w_offset[tail - index] )
            y_offset_new_3[4]=(y_offset_new_3[3]+y_offset_new_3[5])/2
            y_offset_new_3 = y_offset_new_3.permute(1, 0, 2, 3).to(self.device)
            y_new_3 = y_new_3.add(y_offset_new_3)
            
            x_new_3 = x_new_3.sub(y_offset_new_3)
        
        y_new_3 = y_new_3.reshape(
            [self.num_batch, 1, self.num_points, self.w_k, self.h_k])
        
        y_new_3 = y_new_3.permute(0, 3, 1, 4, 2)#num_batch,w_k,num_points,h_k,1

        y_new_3 = y_new_3.reshape([
            self.num_batch, 1 * self.w_k, self.num_points * self.h_k
        ])#
        x_new_3 = x_new_3.reshape(
            [self.num_batch, 1, self.num_points, self.w_k, self.h_k])
        x_new_3 = x_new_3.permute(0, 3, 1, 4, 2)
        x_new_3 = x_new_3.reshape([
            self.num_batch, 1 * self.w_k, self.num_points * self.h_k
        ])

        
        return y_new_2, x_new_2,y_new_3, x_new_3

    def _bilinear_interpolate_3D(self, input_feature, y2, x2,morph):
        y2 = y2.reshape([-1]).float()
        x2 = x2.reshape([-1]).float()
        y2 = torch.floor(y2).int()
        x2 = torch.floor(x2).int()

        input_feature_flat = input_feature.flatten()
        input_feature_flat = input_feature_flat.reshape(
            self.num_batch, self.num_channels, self.width, self.height)
        input_feature_flat = input_feature_flat.permute(0, 2, 3, 1)  
        input_feature_flat = input_feature_flat.reshape(-1, self.num_channels) 
        dimension = self.h_k * self.w_k  

        base = torch.arange(self.num_batch) * dimension
        base = base.reshape([-1, 1]).float() 
        repeat = torch.ones([self.num_points * self.w_k * self.h_k
                             ]).unsqueeze(0)
        repeat = repeat.float() 
        base = torch.matmul(base, repeat) 
        base = base.reshape([-1]) 
        base = base.to(self.device)

        base_y2 = base + y2 * self.height
        index_a0 = base_y2 + x2  
        value_a0 = input_feature_flat[index_a0.type(torch.int64)].to(self.device)
        outputs = value_a0

        if morph == 2:
            outputs = outputs.reshape([
                self.num_batch,
                self.num_points * self.w_k,
                1 * self.h_k,
                self.num_channels,
            ])
            outputs = outputs.permute(0, 3, 1, 2)
        else:
            outputs = outputs.reshape([
                self.num_batch,
                1 * self.w_k,
                self.num_points * self.h_k,
                self.num_channels,
            ])
            outputs = outputs.permute(0, 3, 1, 2)
        return outputs


    def deform_conv(self, input, offset, if_offset):
        y2, x2,y3, x3 = self._coordinate_map_3D(offset, if_offset)
        deformed_feature2 = self._bilinear_interpolate_3D(input, y2, x2,2)
        deformed_feature3 = self._bilinear_interpolate_3D(input, y3, x3,3)
        return deformed_feature2,deformed_feature3


# Code for testing the DSConv
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    time_start = time.time()
    A = np.random.rand(1, 3, 512, 512)
    A = A.astype(dtype=np.float32)
    A = torch.from_numpy(A)

    conv0 = DSConv(
        in_ch=3,
        out_ch=10,
        kernel_size=9,
        stride=1,
        padding=4,
        dilation=1,
        groups=1,
        padding_mode='replicate',
        bias=True,
        if_offset=True)
    if torch.cuda.is_available():
        A = A.to(device)
        conv0 = conv0.to(device)
    out1,out2,_ = conv0(A)
    print(out1.shape,out2.shape)


    time_end = time.time()
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print(time_sum)
    summary(conv0,input_size=(3,512,512))