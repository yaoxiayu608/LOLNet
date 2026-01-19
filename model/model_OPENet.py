import torch
from torch.nn.functional import grid_sample, adaptive_avg_pool2d
from torch import nn, Tensor
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.nn import BatchNorm1d
from torch.nn import BatchNorm2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
# from BinOp import BinOp
from .loss import get_smooth_loss, noOPAL, Lossv4, get_distillation_loss
from .basemodel import BaseModel, init_net, get_optimizer
# from .layers import *
from .context_adjustment_layer import *
import time
import os
import brevitas.nn as qnn
from brevitas.nn import QuantConv2d, QuantIdentity, QuantLinear, QuantReLU, QuantHardTanh, QuantTanh, QuantSigmoid
from brevitas.quant.scaled_int import Int8ActPerTensorFloat,Int8WeightPerChannelFloat,Uint8ActPerTensorFloatMaxInit
from brevitas.quant import SignedBinaryWeightPerTensorConst
from brevitas.quant.solver import ActQuantSolver,WeightQuantSolver
from brevitas.quant.base import SignedBinaryClampedConst
from brevitas.quant import Uint8ActPerTensorFloatMaxInit, Int8ActPerTensorFloatMinMaxInit
from brevitas.quant import Int8WeightPerChannelFloat
from brevitas.quant import Int8WeightPerTensorFloat,Uint8ActPerTensorFloat
from brevitas.quant import IntBias
from brevitas.quant import TruncTo8bit
from brevitas.quant_tensor import QuantTensor
from brevitas.core.stats import AbsPercentile
from brevitas.core.function_wrapper import TensorClamp, OverTensorView, RoundSte, InplaceTensorClampSte
from brevitas.core.scaling import ParameterFromRuntimeStatsScaling, IntScaling
from brevitas.core.stats import AbsPercentile
from brevitas.core.quant import BinaryQuant,ClampedBinaryQuant,QuantType,IntQuant,RescalingIntQuant
from brevitas.core.scaling import ConstScaling,ParameterScaling,ParameterFromStatsFromParameterScaling
from brevitas.core.bit_width import BitWidthImplType,BitWidthConst
from brevitas.core.restrict_val import FloatToIntImplType,RestrictValueType,FloatRestrictValue
from brevitas.core.scaling import ScalingImplType
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.proxy import WeightQuantProxyFromInjector,ActQuantProxyFromInjector
from brevitas.inject.enum import *
from brevitas.inject import ExtendedInjector,value,this
from typing import List
from dependencies import value


def default_conv(in_channels, out_channels, kernel_size, bias= False):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))

    
class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBnReLU(in_channels, out_channels, kernel_size=3, stride=stride, pad=1)
        self.conv2 = ConvBn(out_channels, out_channels, kernel_size=3, stride=1, pad=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


class conv_2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(conv_2d, self).__init__()
        pad = kernel_size // 2 
        # self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad), nn.BatchNorm2d(out_channels))
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        return self.conv(x) 

    
class conv_3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(conv_3d, self).__init__()
        pad = kernel_size // 2 
        block = []
        block.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride, pad))
        block.append(nn.BatchNorm3d(out_channels))
        self.conv = nn.Sequential(*block)

    def forward(self, x):
        return self.conv(x) 

    
class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,  dilation,  downsample=None):
        super(conv_block, self).__init__()
        # pad = kernel_size // 2 
        
        self.downsample = downsample
        block = nn.ModuleList()
        block.append(nn.Conv2d(in_channels, out_channels, 3,1,1, dilation=dilation))
        block.append(nn.BatchNorm2d(out_channels))
        block.append(nn.LeakyReLU(0.2, True))
        block.append(nn.Conv2d(out_channels, out_channels, 3,1,1, dilation=dilation))
        block.append(nn.BatchNorm2d(out_channels))
        self.conv = nn.Sequential(*block)

    def forward(self, x):
        x_skip = x
        if self.downsample is not None:
            x_skip = self.downsample(x)
        return self.conv(x) + x_skip


class Feature(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Feature, self).__init__()
        pad = kernel_size // 2 
        self.relu = nn.LeakyReLU(0.2, True)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 4, 3, 1,1), nn.LeakyReLU(0.2, True))
        
        self.layer1 = self._make_layer(4, 4, 2, 1, 1)
        self.layer2 = self._make_layer(4, 8, 2, 1, 1)
        self.layer3 = self._make_layer(8, 16, 2, 1, 1)
       

        self.branch1 = nn.Sequential(nn.AvgPool2d(2,2), conv_2d(16, 4,1,1), nn.LeakyReLU(0.2, True), nn.UpsamplingBilinear2d(scale_factor=2))
        self.branch2 = nn.Sequential(nn.AvgPool2d(4,4), conv_2d(16, 4,1,1), nn.LeakyReLU(0.2, True), nn.UpsamplingBilinear2d(scale_factor=4))
        self.branch3 = nn.Sequential(nn.AvgPool2d(8,8), conv_2d(16, 4,1,1), nn.LeakyReLU(0.2, True), nn.UpsamplingBilinear2d(scale_factor=8))
       
        self.lastconv = nn.Sequential(conv_2d(28, 16, 3, 1), nn.LeakyReLU(0.2, True), nn.Conv2d(16,out_channels,1,1))

    def _make_layer(self, in_c, out_c, blocks, stride, dilation):
        downsample = None
        if stride != 1 or in_c != out_c:
            downsample = conv_2d(in_c, out_c,1,1) 
        
        layers = []
        layers.append(conv_block(in_c, out_c, 3, stride, dilation, downsample))
        for _ in range(1, blocks):
            layers.append(conv_block(out_c, out_c, 3, stride, dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        
        x = torch.cat([l3, self.branch1(l3), self.branch2(l3), self.branch3(l3)], 1)
        x = self.lastconv(x)
        return x


class CommonQuant(ExtendedInjector):
    bit_width_impl_type = BitWidthImplType.CONST
    scaling_impl_type = ScalingImplType.CONST
    restrict_scaling_type = RestrictValueType.FP
    zero_point_impl = ZeroZeroPoint
    float_to_int_impl_type = FloatToIntImplType.ROUND
    scaling_per_output_channel = False
    narrow_range = True
    signed = True

    @value
    def quant_type(bit_width):
        if bit_width is None:
            return QuantType.FP
        elif bit_width == 1:
            return QuantType.BINARY
        else:
            return QuantType.INT


class CommonWeightQuant(CommonQuant, WeightQuantSolver):
    # scaling_per_output_channel = True
    # @value
    # def scaling_const(module):
    #     negMean = module.weight.mean(1, keepdim=True).mul(-1).expand_as(module.weight)
    #     num_pixel = module.weight[0].nelement()
    #     a = module.weight.add(negMean).clamp(-1.0, 1.0).norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(num_pixel)
    #     #a = module.weight.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(num_pixel)
    #     # print(a[0:3])
    #     return a
    scaling_const = 1.0


class CommonActQuant(CommonQuant, ActQuantSolver):
    min_val = -1.0
    max_val = 1.0


# PerTensor指的其实是PerLayer
# PerChannel指的是逐输出通道
class CommonIntWeightPerTensorQuant(Int8WeightPerTensorFloat):
    """
    Common per-tensor weight quantizer with bit-width set to None so that it's forced to be
    specified by each layer.
    """
    scaling_min_val = 2e-16
    bit_width = None


class CommonIntWeightPerChannelQuant(CommonIntWeightPerTensorQuant):
    """
    Common per-channel weight quantizer with bit-width set to None so that it's forced to be
    specified by each layer.
    """
    scaling_per_output_channel = True


class CommonIntActQuant(Int8ActPerTensorFloatMinMaxInit):
    """
    Common signed act quantizer with bit-width set to None so that it's forced to be specified by
    each layer.
    """
    scaling_min_val = 2e-16
    bit_width = None
    min_val = -10.0
    max_val = 10.0
    restrict_scaling_type = RestrictValueType.LOG_FP


class CommonUintActQuant(Uint8ActPerTensorFloatMaxInit):
    """
    Common unsigned act quantizer with bit-width set to None so that it's forced to be specified by
    each layer.
    """
    scaling_min_val = 2e-16
    bit_width = None
    max_val = 6.0
    restrict_scaling_type = RestrictValueType.LOG_FP

# class BuildingBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, groups, bias, weight_bit_width, act_bit_width):
#         super(BuildingBlock, self).__init__()

#         weight_quant = CommonWeightQuant
#         act_quant = CommonActQuant 

    
# Dropout是一种正则化技术，能够防止模型过拟合，提高模型的泛化能力。
# 在训练时，Dropout 会随机丢弃部分神经元；在测试时，Dropout 不会丢弃任何神经元，但会将输出乘以保留概率（1 - p）以保持期望输出不变。
class BuildingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, groups, bias, weight_bit_width, act_bit_width):
        super(BuildingBlock, self).__init__()

        weight_quant = CommonIntWeightPerChannelQuant
        act_quant = CommonUintActQuant 
                        
        self.conv = QuantConv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride = stride,
                    dilation = dilation,
                    padding = padding,
                    groups = groups,
                    bias=False,
                    weight_quant=weight_quant,
                    weight_bit_width=weight_bit_width)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = QuantReLU(
                                act_quant=act_quant,
                                bit_width=act_bit_width,
                                per_channel_broadcastable_shape=(1, out_channels, 1, 1),
                                scaling_per_channel=False)
        # self.dropout = nn.Dropout(p=0.05)  # p表示随机丢弃对应比列的神经元
    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output) 
        # output = self.dropout(output) 
        return output

    
# 临时的（使用完之后删掉）
class resual_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, groups, bias, weight_bit_width, act_bit_width):
        super(resual_Block, self).__init__()

        weight_quant = CommonIntWeightPerChannelQuant
        act_quant = CommonUintActQuant 
                        
        self.conv1 = QuantConv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride = stride,
                    dilation = dilation,
                    padding = padding,
                    groups = groups,
                    bias=False,
                    weight_quant=weight_quant,
                    weight_bit_width=weight_bit_width)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = QuantReLU(
                                act_quant=act_quant,
                                bit_width=act_bit_width,
                                per_channel_broadcastable_shape=(1, out_channels, 1, 1),
                                scaling_per_channel=False)
        self.conv2 = QuantConv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride = stride,
                    dilation = dilation,
                    padding = padding,
                    groups = groups,
                    bias=False,
                    weight_quant=weight_quant,
                    weight_bit_width=weight_bit_width)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = QuantReLU(
                                act_quant=act_quant,
                                bit_width=act_bit_width,
                                per_channel_broadcastable_shape=(1, out_channels, 1, 1),
                                scaling_per_channel=False)
        
        # self.dropout = nn.Dropout(p=0.05)  # p表示随机丢弃对应比列的神经元
    def forward(self, input):
        output = self.conv1(input)
        
        input_mid1 = output + input
        input_mid1 = self.bn1(input_mid1)
        output = self.relu1(input_mid1) 
        input_mid2 = self.conv2(output)
        
        output = output + input_mid2
        output = self.bn2(output)
        output = self.relu2(output)
        # output = self.dropout(output) 
        return output

    
    
## 深度可分离的卷积操作
class DSNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, bias, weight_bit_width, act_bit_width):
        super(DSNetBlock, self).__init__()
        
        self.deepconv = BuildingBlock(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=kernel_size,
                            stride = stride,
                            dilation = dilation,
                            padding = padding,
                            groups = in_channels,
                            bias=False,
                            weight_bit_width=weight_bit_width,
                            act_bit_width=act_bit_width)
        self.pointconv = BuildingBlock(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=1,
                            stride = 1,
                            dilation = 1,
                            padding = 0,
                            groups = 1,
                            bias=False,
                            weight_bit_width=weight_bit_width,
                            act_bit_width=act_bit_width)
    def forward(self, input):
        output = self.deepconv(input)
        output = self.pointconv(output) 
        return output
    
    
IDX = [[36+j for j in range(9)],
       [4+9*j for j in range(9)],
       [10*j for j in range(9)], 
       [8*(j+1) for j in range(9)]]

index = [36,37,38,39,40,41,42,43,44,4,13,22,31,40,49,58,67,76,0,10,20,30,40,50,60,70,80,8,16,24,32,40,48,56,64,72]


class OPENetmodel(BaseModel): # Basemodel
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['L1'] 
        if opt.losses.find('smooth') != -1:
            self.loss_names.append('smoothness')
        self.visual_names = ['center_input', 'output','label']
        self.model_names = ['EPI']
        net = eval(self.opt.net_version)(opt, self.device,self.isTrain)
        self.netEPI = init_net(net, opt.init_type, opt.init_gain, self.gpu_ids )                           
        self.use_v = opt.use_views

        
        
        # 在 Python 的类中，self 是一个指向类实例本身的引用。使用 self. 定义变量时，这些变量会成为类实例的属性，而不是局部变量。这意味着这些变量可以在类的任何方法中访问，并且会随着实例的生命周期而存在。
        # 如果不使用 self. 定义变量，这些变量将被视为局部变量，仅在定义它们的方法内部有效。它们不会成为类实例的属性，因此无法在类的其他方法中访问。
        # 加载预训练的教师模型
        self.teacher_model = TeacherModel(self.device)
        save_dir = os.path.join('./checkpoints', 'OPENet', '2025-04-04-14-50')
        load_filename = '%s_net_%s.pth' % ('204', 'EPI')  # 教师模型的MSE：2.257
        load_path = os.path.join(save_dir,load_filename)
        self.teacher_model.load_state_dict(torch.load(load_path, map_location=str(self.device)))  # 加载权重文件
        self.teacher_model.to(self.device)
        # # 冻结教师模型的参数（这段代码加不加对效果没有影响）
        # # for param in self.teacher_model.parameters():
        # #     param.requires_grad = False  # 用于训练阶段，冻结参数，使其不更新。
        # self.teacher_model.eval()  # 将教师模型设置为评估模式

        
        
        self.center_index = self.use_v // 2
        self.alpha = opt.alpha
        self.pad = opt.pad
        self.lamda = opt.lamda

        self.test_loss_log = 0
        self.test_loss = torch.nn.L1Loss()
        if self.isTrain:
            # define loss functions
            self.criterionL1 = eval(self.opt.loss_version)(self.opt,self.device)
            self.optimizer = get_optimizer(opt, filter(lambda p: p.requires_grad, self.netEPI.parameters()), LR=opt.lr)
            self.optimizers.append(self.optimizer)
        
    def set_input(self, inputs, epoch):
        self.epoch = epoch
        # self.supervise_view = rearrange(inputs[0].to(self.device), 'b c (h1 h) (w1 w) u v -> (b h1 w1) c h w u v', h1=8, w1=8)
        self.supervise_view = inputs[0].to(self.device)
        self.input = []
        for j in range(self.use_v):
            self.input.append(self.supervise_view[:,:,:,:, self.center_index,j])
        for j in range(self.use_v):
            self.input.append(self.supervise_view[:,:,:,:, j,self.center_index])    
        for j in range(self.use_v):
            self.input.append(self.supervise_view[:,:,:,:, j,j]) 
        for j in range(self.use_v):
            self.input.append(self.supervise_view[:,:,:,:, j,self.use_v-1-j])    
            
        # image = self.supervise_view
        # image = image*255
        # crop_size = 48
        # _, _, h, w, _, _ = image.shape
        # if h < crop_size or w < crop_size:
        #     raise ValueError("Image size is too small for cropping.")
        # start_x = 200
        # start_y = 400
        # # 裁剪图像
        # image = image[:,:,start_y:start_y+crop_size, start_x:start_x+crop_size,:,:]
        # np.save("input_9*9*48*48.npy", image.cpu())
            
            
        # self.input_5 = []
        # for j in range(2,7):
        #     self.input_5.append(self.supervise_view[:,:,:,:, self.center_index,j])
        # for j in range(2,7):
        #     self.input_5.append(self.supervise_view[:,:,:,:, j,self.center_index])    
        # for j in range(2,7):
        #     self.input_5.append(self.supervise_view[:,:,:,:, j,j]) 
        # for j in range(2,7):
        #     self.input_5.append(self.supervise_view[:,:,:,:, j,self.use_v-1-j])  
        self.input_7 = []
        for j in range(1,8):
            self.input_7.append(self.supervise_view[:,:,:,:, self.center_index,j])
        for j in range(1,8):
            self.input_7.append(self.supervise_view[:,:,:,:, j,self.center_index])     
        for j in range(1,8):
            self.input_7.append(self.supervise_view[:,:,:,:, j,j]) 
        for j in range(1,8):
            self.input_7.append(self.supervise_view[:,:,:,:, j,self.use_v-1-j])  
        # self.input_33 = []
        # for i,x in enumerate(self.input):
        #     if i not in (4, 13, 21):
        #         self.input_33.append(x)
        self.gray_input = []
        for map in self.input_7:
            map = map[:,0,:,:]*0.299+map[:,1,:,:]*0.587+map[:,2,:,:]*0.114
            self.gray_input.append(map.unsqueeze(1))
        # self.full_input = []
        # for i in range(1,8):
        #     for j in range(1,8):
        #         self.full_input.append(self.supervise_view[:,:,:,:, i,j]) 
        # for i in range(0,9):
        #     for j in range(0,9):
        #         self.full_input.append(self.supervise_view[:,:,:,:, i,j]) 
        self.center_input = self.input[self.center_index]
        self.label = inputs[1].to(self.device)
        
    def forward(self,isTrain):#############################################################################
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.output, self.raw_warp_img = self.netEPI(self.gray_input)
        self.test_loss_log = 0
        # 获取教师模型的输出(训练的时候用于蒸馏，测试的时候注释掉教师模型)
        if isTrain:
            with torch.no_grad():  # 用于推理阶段，禁用梯度计算，减少显存占用并提高计算效率。
                self.teacher_outputs = self.teacher_model(self.input)

    def backward_G(self):
        # if self.epoch <= self.opt.n_epochs:
        #     self.loss_L1 = self.criterionL1(self.epoch,self.output, self.input[:9], self.input[9:18])
        # else:
        #     self.loss_L1 = 0
        # self.loss_L1 = self.criterionL1(self.output, self.input[:9], self.input[9:18])
        # self.loss_L1 = self.criterionL1(self.epoch,self.output,self.supervise_view,self.center_input)
        self.loss_L1 = self.criterionL1(self.epoch,self.output, self.input[:9], self.input[9:18], self.input[18:27], self.input[27:])
        # self.loss_L1 = self.criterionL1(self.output, self.full_input)
        if self.raw_warp_img is None:
             self.loss_total = self.loss_L1
        else:
            self.loss_raw = 0
            for views in self.raw_warp_img:
                self.loss_raw += self.criterionL1(self.epoch,views)  # 
            self.loss_total = 0.6*self.loss_L1 + 0.4*self.loss_raw
            # self.loss_total = self.loss_L1
        self.loss_smoothness = get_smooth_loss(self.output, self.center_input, self.lamda) 
        # 计算蒸馏损失
        self.loss_distillation = get_distillation_loss(self.output, self.teacher_outputs, self.epoch,0.5)
        # 蒸馏损失比光度一致性损失小一个数量级
        # print('loss_total',self.loss_total)
        # print('loss_distillation',self.loss_distillation)
        self.loss_total += self.loss_distillation 
        # 该损失与学习率衰减周期n_epochs有关，但是无论n_epochs设置为多少，一旦加入平滑损失，低比特网络的效果立即变得糟糕
        # if 'smoothness' in self.loss_names and self.epoch > 2*self.opt.n_epochs:
        #     self.loss_total += self.loss_smoothness 
        self.loss_total.backward()
        
    def optimize_parameters(self):
        self.netEPI.train()
        self.forward(isTrain=True)                   
        self.optimizer.zero_grad()       
        self.backward_G()                   
        self.optimizer.step()           

    
class TeacherModel(nn.Module):
    def __init__(self,device):
        super().__init__()
        feats = 64
        self.device = device
        self.use_v = 9
        self.feat_extract = Feature(in_channels=3, out_channels=8)
        self.block3d = nn.ModuleList()
        self.fuse3d = nn.ModuleList()
        for _ in range(4):
            self.block3d.append(
                nn.Sequential(nn.Conv3d(8, 64, (self.use_v, 3, 3), 1, padding=(0, 1, 1)), nn.LeakyReLU(0.2, True)))
            self.fuse3d.append(nn.Sequential(nn.Conv2d(64, 32, 1, 1), nn.LeakyReLU(0.2, True)))
        feats *= 2
        self.fuse2d = nn.ModuleList()
        for j in range(4):
            self.fuse2d.append(nn.Sequential(nn.Conv2d(feats, feats, 3, 1, padding=1), nn.LeakyReLU(0.2, True),
                                             nn.Conv2d(feats, feats, 1, 1)))
            self.fuse2d.append(nn.LeakyReLU(0.2, True))
        self.fuse3 = nn.Conv2d(feats, 9, 3, 1, padding=1)
        self.relu3 = nn.Softmax(dim=1)
        self.finetune = ContextAdjustmentLayerv2()

    def forward(self, x):
        feats = []
        for xi in x:
            feat_i = self.feat_extract(xi)
            feats.append(feat_i)
        feats_angle = []
        for j in range(4):
            feats_tmp = torch.stack(feats[self.use_v * j:self.use_v * (j + 1)], dim=2)
            feats_tmp = self.block3d[j](feats_tmp)
            feats_tmp = torch.squeeze(feats_tmp, 2)
            feats_angle.append(self.fuse3d[j](feats_tmp))
        cv = torch.cat(feats_angle, 1)
        for j in range(4):
            cv = self.fuse2d[j * 2 + 1](self.fuse2d[j * 2](cv) + cv)
        prob = self.relu3(self.fuse3(cv))
        disp_raw = self.disparitygression(prob)
        disp_final = self.finetune(disp_raw, x[4])

        return disp_final

    def disparitygression(self, input):
        disparity_values = torch.linspace(-4, 4, 9, device=self.device)
        x = disparity_values.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        out = torch.sum(torch.multiply(input, x), 1)
        return out.unsqueeze(1)
    
    
class AdjustmentLayerv2(nn.Module):
    def __init__(self, num_blocks=8, feature_dim=16, expansion=3):
        super().__init__()
        self.num_blocks = num_blocks

        self.in_conv = QuantConv2d(
                    in_channels=2,
                    out_channels=feature_dim,
                    kernel_size=3,
                    stride = 1,
                    dilation = 1,
                    padding = 1,
                    groups = 1,
                    bias=False,
                    weight_quant=CommonIntWeightPerChannelQuant,
                    weight_bit_width=4)
        self.layers = nn.ModuleList([ResBlock(feature_dim, expansion) for _ in range(num_blocks)])
        self.out_conv = QuantConv2d(
                    in_channels=feature_dim,
                    out_channels=1,
                    kernel_size=3,
                    stride = 1,
                    dilation = 1,
                    padding = 1,
                    groups = 1,
                    bias=False,
                    weight_quant=CommonIntWeightPerChannelQuant,
                    weight_bit_width=8)

    def forward(self, disp_raw: Tensor,  img: Tensor):

        feat = self.in_conv(torch.cat([disp_raw, img], dim=1))
        for layer in self.layers:
            feat = layer(feat, disp_raw)
        disp_res = self.out_conv(feat)
        disp_final = disp_raw + disp_res
 
        return disp_final

class ResBlock(nn.Module):
    def __init__(self, n_feats: int, expansion_ratio: int, res_scale: int = 1.0):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.module = nn.Sequential(
            weight_norm(QuantConv2d(
                    in_channels=n_feats + 1,
                    out_channels=n_feats * expansion_ratio,
                    kernel_size=3,
                    stride = 1,
                    dilation = 1,
                    padding = 1,
                    groups = 1,
                    bias=False,
                    weight_quant=CommonIntWeightPerChannelQuant,
                    weight_bit_width=4)),
            QuantReLU(act_quant=CommonUintActQuant,bit_width=4,per_channel_broadcastable_shape=(1, n_feats * expansion_ratio, 1, 1),scaling_per_channel=False),
            weight_norm(QuantConv2d(
                    in_channels=n_feats * expansion_ratio,
                    out_channels=n_feats,
                    kernel_size=3,
                    stride = 1,
                    dilation = 1,
                    padding = 1,
                    groups = 1,
                    bias=False,
                    weight_quant=CommonIntWeightPerChannelQuant,
                    weight_bit_width=4))
        )

    def forward(self, x: torch.Tensor, disp: torch.Tensor):
        return x + self.module(torch.cat([disp, x], dim=1)) * self.res_scale

# class AdjustmentLayerv2(nn.Module):
#     def __init__(self, num_blocks=8, feature_dim=16, expansion=3):
#         super().__init__()
#         self.num_blocks = num_blocks

#         self.in_conv = nn.Conv2d(2, feature_dim, kernel_size=3, padding=1)
#         self.layers = nn.ModuleList([ResBlock(feature_dim, expansion) for _ in range(num_blocks)])
#         self.out_conv = nn.Conv2d(feature_dim, 1, kernel_size=3, padding=1)

#     def forward(self, disp_raw: Tensor,  img: Tensor):

#         feat = self.in_conv(torch.cat([disp_raw, img], dim=1))
#         for layer in self.layers:
#             feat = layer(feat, disp_raw)
#         disp_res = self.out_conv(feat)
#         disp_final = disp_raw + disp_res
 

#         return disp_final

# class ResBlock(nn.Module):
#     def __init__(self, n_feats: int, expansion_ratio: int, res_scale: int = 1.0):
#         super(ResBlock, self).__init__()
#         self.res_scale = res_scale
#         self.module = nn.Sequential(
#             weight_norm(nn.Conv2d(n_feats + 1, n_feats * expansion_ratio, kernel_size=3, padding=1)),
#             nn.ReLU(inplace=True),
#             weight_norm(nn.Conv2d(n_feats * expansion_ratio, n_feats, kernel_size=3, padding=1))
#         )

#     def forward(self, x: torch.Tensor, disp: torch.Tensor):
#         return x + self.module(torch.cat([disp, x], dim=1)) * self.res_scale
    

class Unsup27_16_16(nn.Module):  # v3
    def __init__(self,opt,device, is_train=True):
        super().__init__() 
        self.is_train = is_train
        self.device = device
        self.use_v = opt.use_views
        self.grad_v = opt.grad_v
        self.weight_bit_width = 4
        self.act_bit_width = 4
        self.channel_number = 64
    
        # LOLNet (FP)
        # self.conv1 = ConvBnReLU(28, 64, kernel_size=3, stride=1, pad=1)
        # self.conv2 = ConvBnReLU(64, 64, kernel_size=3, stride=1, pad=1)
        # self.conv3 = ConvBnReLU(64, 64, kernel_size=3, stride=1, pad=1)
        # self.conv6 = ConvBnReLU(64, 64, kernel_size=3, stride=1, pad=1)
        # self.conv7 = ConvBnReLU(64, 50, kernel_size=3, stride=1, pad=1)
        # self.conv4 = nn.Conv2d(50, 9, 3, stride=1, padding=1, bias=False)
        
        
        # LOLNet
        self.conv1 = BuildingBlock(in_channels=28, out_channels=self.channel_number, kernel_size=3, stride=1, dilation=1, padding=1, 
                          groups=1, bias=False, weight_bit_width=self.weight_bit_width, act_bit_width=self.act_bit_width)
        # self.conv2 = BuildingBlock(in_channels=self.channel_number, out_channels=self.channel_number, kernel_size=3, stride=1, dilation=1, padding=1, 
        #                   groups=1, bias=False, weight_bit_width=self.weight_bit_width, act_bit_width=self.act_bit_width)
        # self.conv3 = BuildingBlock(in_channels=self.channel_number, out_channels=self.channel_number, kernel_size=3, stride=1, dilation=1, padding=1, 
        #                   groups=1, bias=False, weight_bit_width=self.weight_bit_width, act_bit_width=self.act_bit_width)
        self.resual_Block = resual_Block(in_channels=self.channel_number, out_channels=self.channel_number, kernel_size=3, stride=1, dilation=1, padding=1, 
                           groups=1, bias=False, weight_bit_width=self.weight_bit_width, act_bit_width=self.act_bit_width)
        self.resual_Block1 = resual_Block(in_channels=self.channel_number, out_channels=self.channel_number, kernel_size=3, stride=1, dilation=1, padding=1, 
                           groups=1, bias=False, weight_bit_width=self.weight_bit_width, act_bit_width=self.act_bit_width)
        # self.conv6 = BuildingBlock(in_channels=self.channel_number, out_channels=self.channel_number, kernel_size=3, stride=1, dilation=1, padding=1, 
        #                   groups=1, bias=False, weight_bit_width=self.weight_bit_width, act_bit_width=self.act_bit_width)
        self.conv7 = BuildingBlock(in_channels=self.channel_number, out_channels=50, kernel_size=3, stride=1, dilation=1, padding=1, 
                          groups=1, bias=False, weight_bit_width=self.weight_bit_width, act_bit_width=self.act_bit_width)
        
        # 深度可分离卷积
        # self.conv1 = DSNetBlock(in_channels=28, out_channels=32, kernel_size=3, stride=1, dilation=1, padding=1, 
        #                    bias=False, weight_bit_width=4, act_bit_width=4)
        # # self.conv2 = DSNetBlock(in_channels=180, out_channels=180, kernel_size=3, stride=1, dilation=1, padding=1, 
        # #                    bias=False, weight_bit_width=4, act_bit_width=4)
        # # self.conv3 = DSNetBlock(in_channels=180, out_channels=180, kernel_size=3, stride=1, dilation=1, padding=1, 
        # #                    bias=False, weight_bit_width=4, act_bit_width=4)
        # # self.conv6 = DSNetBlock(in_channels=180, out_channels=180, kernel_size=3, stride=1, dilation=1, padding=1, 
        # #                    bias=False, weight_bit_width=4, act_bit_width=4)
        # # self.conv7 = DSNetBlock(in_channels=180, out_channels=180, kernel_size=3, stride=1, dilation=1, padding=1, 
        # #                    bias=False, weight_bit_width=4, act_bit_width=4)
        # self.conv4 = QuantConv2d(
        #             in_channels=32,
        #             out_channels=9,
        #             kernel_size=3,
        #             stride = 1,
        #             dilation = 1,
        #             padding = 1,
        #             groups = 1,
        #             bias=False,
        #             weight_quant=CommonIntWeightPerChannelQuant,
        #             weight_bit_width=4)
        # layers_feature = []
        # for _ in range(8):
        #     layers_feature.append(DSNetBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, dilation=1, padding=1, 
        #                    bias=False, weight_bit_width=4, act_bit_width=4))
        # self.init_feature = nn.Sequential(*layers_feature)
        
        self.conv4 = QuantConv2d(
                    in_channels=50,
                    out_channels=9,
                    kernel_size=3,
                    stride = 1,
                    dilation = 1,
                    padding = 1,
                    groups = 1,
                    bias=False,
                    weight_quant=CommonIntWeightPerChannelQuant,
                    weight_bit_width=8)
        # self.relu = QuantReLU(
        #                         act_quant=CommonUintActQuant,
        #                         bit_width=8,
        #                         per_channel_broadcastable_shape=(1, 9, 1, 1),
        #                         scaling_per_channel=False)

        self.relu3 = nn.Softmax(dim=1)
        # self.finetune = AdjustmentLayerv2()

        # 消融实验：
        # Ours w/o AOM：2.431 17.49 第4761个epoch 2025-07-22-03-06
        # Ours w/o KD：4.578 17.68 第4809个epoch 2025-07-25-12-56
        # Ours w/o KD and AOM：8.097 22.45 第4650个epoch 2025-07-23-02-39
        # Ours：2.234 16.59 第4947个epoch 2025-07-22-01-42
        # 有没有一种策略，就是按step去衰减，但是衰减了4次之后又回到最初的学习率继续按step衰减，多循环几次，这样做的目的是为了用大的学习率跳出当前的局部最优点
        # 使用CosineAnnealingWarmRestarts(optimizer, T_0=opt.lr_decay_iters, T_mult=1, eta_min=5.67e-6)可以达到类似的效果
        
    def forward(self, x): 
        # 这段代码用于测试训练的效果，也用于生成上板数据（从512*512的图片中随机裁剪出64*64的图片）（将9个视差融在了batch维度）
        # B,C,H,W = x[0].shape
        # predefined_disp = []
        # shape = (B, 1, H, W)
        # fill_value = -5
        # for i in range(9):
        #     fill_value+=1
        #     tensor = torch.full(shape, fill_value, dtype=torch.float32, device='cpu')
        #     predefined_disp.append(tensor)
        # feats = []
        # inputs = []  # 创建一个空列表
        # for i in range(9):
        #     feature = self.shift(predefined_disp[i],x)
        #     cv = torch.cat(feature,1)
        #     print('cv.shape:',cv.shape)  # [1, 36, 512, 512]
        #     inputs.append(cv)  # 将每个视差下经过warp之后的图片放入该列表
        #     cv = self.conv1(cv)
        #     cv = self.conv2(cv)
        #     cv = self.conv3(cv)
        #     cv = self.conv4(cv)
        #     feats.append(cv)
        # image = torch.cat(inputs,0)  # 将视差维度融在batch维度
        # print('image.shape:',image.shape)  # [9, 36, 512, 512]
        # print('original dtype:',image.dtype)  # torch.float32
        # image = image.type(torch.uint8)
        # np.save("input2.npy", image.cpu())
        # crop_size = 64
        # _, _, h, w = image.shape
        # # 确保图像尺寸足够大以进行裁剪
        # if h < crop_size or w < crop_size:
        #     raise ValueError("Image size is too small for cropping.")
        # # 随机选择起始点
        # start_x = np.random.randint(0, w - crop_size)
        # start_y = np.random.randint(0, h - crop_size)
        # # 裁剪图像
        # image = image[:,:,start_y:start_y+crop_size, start_x:start_x+crop_size]
        # print('final dtype:',image.dtype)  # torch.uint8
        # print('final shape',image.shape)  # [9, 36, 64, 64]
        # np.save("input.npy", image.permute(0, 2, 3, 1).cpu())
        # disp_raw = torch.cat(feats,1)
        # prob = self.relu3(disp_raw)
        # disp_raw = self.disparitygression(prob)
        
        # 三大点：
        # 1.调整超参数，如学习率，epoch数（要合适，太大太小都不行）（设置dropout）（52.6790，这么大的MSE表明网络参数没有进行更新，原因是简单的网络架构在堆叠7层卷积后将导致梯度消失）
        # 2.损失函数：去掉平滑损失（再试试视差划分前置的方法）（使用预训练权重）（也可以考虑加入平滑损失试试）（平滑损失对噪点没多大影响）
        # 3.网络架构（层数、通道数）以及加入类似于BN之类的方法（batch_size=2），输入是4*9=36张视图（输入和损失中所使用的视图数量可以任意设置，灰度图），最后一层权重进行8bit量化（没有视差回归也可以得到视差图）（3*3和1*1卷积交替的最后一个卷积必须为3*3卷积，否则效果不好）（1*1卷积有一定的效果，而且增加的参数量不多，但是会带来更多的计算量）（3*3和1*1卷积交替的效果并不是很突出）（增加通道数不如增加一个卷积层的效果好）（增加卷积层数但是减少通道数的效果较好）
        # 没有残差连接的话，信息会丢失很多，之后堆再多的卷积也没有用
        # 量化会丢失很多信息，做完卷积之后反量化可减少部分信息损失。量化之后的定点比量化前的浮点计算简单，且结果不会差太多，因为提了一个缩放因子出来：ax*w=a(x*w)
        # 奇异值分解中重要的部分使用高比特，不重要的使用低比特（哈夫曼编码）
        # 量化分为量化后训练和训练后量化，但训练后量化精度会掉很多，我们采取量化后训练。
        # PTQ是如何实现的（用8bit训练好参数之后进行4bit测试，不会报错，似乎这就是训练后量化，不过精度大幅降低），可以试着用高bit权重作为预训练模型然后训练低bit网络
        # 生成对抗网络
        # 直接用伪量化训练好的权重做测试
        
        
        # 在9090里的Jupyter中的代码就是PS端
        # 在driver.py中实现的pipeline
        # 池化：大图作为输入，小图作为输出
        # 增加损失，用中间输出去约束
        # 开启异步，.xpr文件，64*64小图拼大图
        # 权重分布可视化，为什么gt效果更好
        # 具身智能、倒车影像、无人驾驶、无人送货车
        # W 1-bit  A 2-bit
        # Note that weights at stage 2 are either 1 or -1,so applying an L2 regularization term to them does nt make sense.

        
        # 噪点的存在可能是因为没有残差导致信息丢失，输入中的一些信息再也无法出现，cv = self.conv2(cv) + cv，self.conv3(cv) + cv，self.conv6(cv) + cv的MSE：2.966(可视化效果也不好)
        # 加入OPAL细化模块的MSE：2.709 第1026个epoch 2025-03-11-08-58(可视化效果很好，噪点消失)
      
        # 微调阈值的两个极端分别是map_min和map_all，当阈值足够大时，相当于不存在遮挡，全部像素点都使用map_all，当阈值足够小时，相当于处处是遮挡，全部像素点都使用map_min
        # loss = map_all*(1-diff)+map_all*diff的MSE：3.028 
        # loss = map_min*(1-diff)+map_min*diff的MSE：2.922 
        # 有遮挡的时候取map_min，没遮挡的时候取map_all（没遮挡的时候不能也取map_min，损失并不是说越小越好，用更多的视图参与损失计算才好，map_min只是代表没遮挡的区域）
        # 当存在单侧遮挡或两侧遮挡不对称的时候，map_min一定来自map_up或者map_down，也可能来自map_circle，而不会是map_all，因为map_all是map_up和map_down的平均值
        
        # 增加两层卷积，最后一层8bit量化,教师模型的MSE：2.257，将n_epochs修改为280,加入L1损失，每个方向只使用7张视图，输入为灰度图，使用2.375作为预训练权重：
        # LOSSV4只在每个方向上考虑了“遮挡只发生在一侧”（map_circle是考虑到了两侧存在不对称遮挡的情况，但简单粗暴地使用map_circle进行损失计算），并未考虑两侧都存在遮挡的情况(即对称遮挡，如boxes场景中的网格角落)，在所有diff<0.01的区域，我们认为是不存在单侧遮挡的（也不存在不对称遮挡），用相邻的多张视图和中心视图相减求平均，如果差值（即map_up或map_down）大于0.015，则认为两侧都存在遮挡（简单粗暴地认为8张视图均存在遮挡），两侧都存在遮挡的地方直接取0或者乘以一个较小的系数（微调的时候使用，因为此时的视差图基本准确）
        # mask = torch.where(map_up<0.01,1,0)，将教师模型的L1损失替换为平滑L1损失的MSE：2.347
        # mask = torch.where(map_up<0.01,1,0)，2.257的教师模型也许不再有指导能力，因此去掉教师模型之后的MSE：3.201
        # 去掉LOSSV4的MSE：2.326（由此可见，遮挡模式感知损失可有可无，教师的影响太大了）
        # 使用LOSSV9的MSE：2.327
        # 使用LOSSV5的MSE：2.336
        # 使用LOSSV3的MSE：2.381
        # mask = torch.where(map<0.01,1,0)  self.loss_total = self.loss_total + self.loss_distillation  直接求平均的MSE：2.293 
        
        # 在3090上用OPAL论文里面的八个场景作为验证的MSE为1.261，虽然达不到论文里的效果，但基本上也算是一个说得过去的结果
        # 加载1.261预训练权重进行微调之后的效果还不如不进行微调，按OPAL论文里面的损失系数进行设置之后的效果也不好，把验证集加入训练集效果也不好
        # 加入transformer之后训练的学生模型对dots场景和真实场景不好（教师在训练的时候，设置batchsize=16，训练集长度为4000，每次训练都能收敛到一个不错的结果）
        # 不加入transformer训练的学生模型对旧HCI数据集效果不好（教师在训练的时候，如果设置batchsize=16，训练集长度为4000，只有dots很糟糕，设置batchsize=2，训练集长度为400，不是每次都能收敛，一旦收敛效果不错）
        # OPAL的最终输出是光滑的，细化之前的视差图也比较光滑，去掉细化模块从头训练的结果就没那么光滑了，说明细化模块起作用了
        # 轻量化网络的结果是粗糙的，全精度的学生模型输出结果也是粗糙的，说明噪点出现的原因并不是量化而是因为缺少残差连接导致的信息丢失（消融实验）
        # 轻量化网络在加入细化模块之后如果损失只考虑最终视差图而不考虑细化之前的视差图，则最终输出是光滑的，但细化之前的视差图是相当糟糕的
        # 如果损失只考虑细化之前的视差图而不考虑最终的视差图，则细化之前的视差图是粗糙的，最终的视差图也是粗糙的
        # 如果两者都考虑，则最终输出是光滑的，但细化之前的视差图还是比较糟糕的
        # 对细化模块进行量化，将其中的所有卷积替换为量化卷积，但是只能验证却不能测试

        
        # 教师模型以2025-02-20-06-33 2.257 282作为预训练权重，使用LOSSV4，训练之后并没有更好的效果
        # 教师模型以2025-04-04-14-50 2.099 282作为预训练权重，在所有非0区域求平均，训练之后并没有更好的效果         5.847 204（只是dots场景不好）  8.39 270（只是dots场景不好）
        # 教师模型以2.049作为预训练权重，q = torch.tensor([0.75]).to(map)，初始和细化都使用米字型损失的MSE：2.01
        # 教师模型以2.049作为预训练权重，q = torch.tensor([0.75]).to(map)，初始和细化都使用米字型损失，并用2025-04-04-14-50 8.39 270生成mask的MSE：2025-05-01-14-32 1.957 315     badpixel 7.565 279 2.144
        # 教师模型以2.049作为预训练权重，q = torch.tensor([0.75]).to(map)，初始和细化都使用米字型损失，并用2025-04-04-14-50 8.39 270生成mask，0.45+0.7的MSE：1.896 311 2025-05-03-14-25 10.45(超参数的设置影响并不大)
        
        # 从头训练教师模型并且batchsize=2而且当epoch<=10的时候使用noOPAL的MSE（初始损失使用米字，最终损失使用十字）：2.427
        # 从头训练教师模型并且batchsize=2而且使用noOPAL的MSE（初始损失使用米字，最终损失使用十字）：5.665
        # 教师模型以2.427作为预训练权重，q = torch.tensor([0.75]).to(map)，初始和细化都使用米字型损失的MSE：2.166
        # 教师模型以2.427作为预训练权重，q = torch.tensor([0.75]).to(map)，初始和细化都使用十字型损失的MSE：2.058 第291个epoch 2025-05-14-08-17

        
        # 5.847的教师模型，2.293的学生模型，使用LOSSV4的MSE：2.132（虽然教师模型的dots场景不好，但是学生模型天然地对dots场景好）
        # 2.01的教师模型，2.132的学生模型，q = torch.tensor([0.75]).to(map)的MSE：2.112 第1149个epoch 2025-04-19-03-19
        # 2.005的教师模型，2.112的学生模型，用教师模型生成mask的MSE：还不如只使用蒸馏损失
        # 1.916的教师模型，2.112的学生模型，用教师模型生成mask，并且光度损失*0.1的MSE：效果也不咋滴


        # 能不能冻结训练好的网络，然后只训练细化模块
        # 有没有人做轻量化的细化研究，细化模块中都是什么函数，看看Beyond的细化模块是怎么实现的
        
        # 能不能用教师去引导学生进行细化
        # 能不能使用可变形卷积去做细化模块，只对想要的像素点进行卷积
        # 大图，中图，小图，超小图的分析，发现流处理器中大图的利用率很低，而超小图的利用率高，类比这个分析一下FINN

        # 找一个简单一点的细化网络把噪点去掉
        # 直接用预训练权重进行训练几个epoch就可以测试自己拍摄的真实场景了

        # 7*7视图是没有问题的，因此可以试着使用加入了transformer的OPAL将7*7作为输入训练
        # 从官网上下载的Stanford数据集有问题，正常情况下u和v的起始坐标在左上角，但是士兵数据集的起始坐标在右下角，即该数据集上下左右都反了，兔子数据集的起始坐标在左下角，即该数据集上下反了
        # 用实验室的光场相机拍摄的数据集也有问题，默认情况下程序是按照文件名的字典序（即字符串顺序）来读取文件，假设一个文件夹中存在两张图片，2.png和10.png，程序会先读取10.png，导致读取的图片不是预期想要的图片，因此在对图片进行命名的时候尽量使用多位，eg.002.png和010.png

        # 无论是379还是368，都不影响训练，会裁剪为64*64，而测试的时候必须是偶数
        
        # 计算量和时间不成正比，比如内存的调入调出都需要时间     
        # OAVC        3.84         -         0.19秒
        # 钩子函数
        # 分析一下bad piexl0.1 0.2 0.5 1每个指标下的百分比，制作统计图，说明为什么效果不好

        # In a QONNX graph, all quantization is represented using Quant, BinaryQuant or Trunc nodes.（Trunc？）
        # The result is a model consisting of a mixture of HW and non-HW layers. （能不能将残差连接放在PS端实现）
        # In the next step the graph is split and the part consisting of HW layers is further processed in the FINN flow. The parent graph containing the non-HW layers remains.
        # generate a Vivado IP Integrator (IPI) design with AXI stream (FIFO) in-out interfaces, which can be integrated onto any Xilinx FPGA as part of a larger system.（能否在Vivado中加入残差）
        # If you are using a neural network with a topology that is substantially different to the FINN end-to-end examples, the simple dataflow build mode below is likely to fail. For those cases, we recommend making a copy of the end-to-end Jupyter notebook as a starting point, visualizing the model at intermediate steps and adding calls to new transformations as needed. Once you have a working flow, you can implement a command line entry for this by using the “advanced mode”.（自定义转换）
        
        # Instead of specifying the folding configuration, you can use the target_fps option in the build configuration to control the degree of parallelization for your network.
        
        # finn.builder.build_dataflow_config.DataflowOutputType.ESTIMATE_REPORTS produces a variety of reports to estimate resource usage and performance without running any synthesis. This can be useful for setting up the parallelization and other hardware configuration:（生成各种分析报告  https://finn.readthedocs.io/en/latest/command_line.html#generated-outputs）
        
        # FINN dataflow builds go through many steps before the bitfile is generated, and the flow may produce erronous models due to bugs or unsupported features. When running new models throught this process it’s a good idea to enable the verification features of the dataflow build. In this way, FINN will use the input you provide to run through the intermediate models, produce some output and compare it against the expected output that you provide.（验证结果  https://finn.readthedocs.io/en/latest/command_line.html#verification-of-intermediate-steps）
        
        # These networks are built end-to-end as part of the FINN integration tests , and the key performance indicators (FPGA resource, frames per second…) are automatically posted to the dashboard below. To implement a new network, you can use the integration test code as a starting point, as well as the relevant Jupyter notebooks.（以集成测试代码作为起点）
        

        
        
        # 测试推理时间
        # start_time = time.time()
        cv = torch.cat(x,1)
        # 以下代码用于生成FPGA的uint8输入（512*512或者64*64），该输入可以是RGB图片，也可以是灰度图
        # 注意在加载测试数据的时候将/255去掉（该操作已经包含在了生成位文件的预处理中）,在加载训练数据的时候将batchsize改为1
        # 注意灰度图本身是用8bit表示的，而Gray=0.299R+0.587G+0.114B只是一个近似的转换公式，转换之后的灰度图是用32bit表示的浮点数，
        # 虽然直接将其强制转换为uint8会存在精度丢失，但是对MSE及可视化效果影响不大，而且减少了参数量
        # cv = cv * 255
        # cv = cv.to(torch.uint8)
        # image = cv
        # print('image.shape:',image.shape)  # [1, 108, 512, 512]
        # print('original dtype:',image.dtype)  # torch.float32
        # # image = image.type(torch.uint8)
        # np.save("input512.npy", image.permute(0, 2, 3, 1).cpu())
        # crop_size = 256
        # _, _, h, w = image.shape
        # # 确保图像尺寸足够大以进行裁剪
        # if h < crop_size or w < crop_size:
        #     raise ValueError("Image size is too small for cropping.")
        # # 随机选择起始点
        # # start_x = np.random.randint(0, w - crop_size)
        # # start_y = np.random.randint(0, h - crop_size)
        # start_x = 200
        # start_y = 200
        # # 裁剪图像
        # image = image[:,:,start_y:start_y+crop_size, start_x:start_x+crop_size]
        # print('final dtype:',image.dtype)  # torch.uint8
        # print('final shape',image.shape)  # [1, 108, 64, 64]
        # np.save("input256.npy", image.permute(0, 2, 3, 1).cpu())
        # # cv = self.conv1(image / 255.)
        # cv = cv / 255.  # 这个除以255后面的小数点对可视化效果没有任何影响，从512*512中裁剪出64*64的可视化效果和之前不一致，因为要把64*64当作一个新的输入去处理，这只是映射模式的问题，输出结果的数值大小是一致的？（边界填充会不会造成影响？在计算损失的时候，边缘会不会差距较大，需不需要裁剪掉边缘？）
        # cv = self.conv5(cv)
        # end_time1 = time.time()
        
        # print(cv.shape)
        
        
        
        # 6层3*3卷积的感受野为13*13，而边缘像素的卷积受到填充的影响，因此裁剪掉边缘13*13的像素
        # C, H, W = cv.shape[-3:]  # 获取输入张量的高度和宽度
        # patch_size=128
        # stride=64
        # pad_size=32
        # cv  = F.pad(cv, (pad_size, pad_size, pad_size, pad_size))
        # # 对输入张量进行滑动窗口操作，将输入张量[N,C,H,W]展开成一个二维矩阵，每一行对应一个窗口的内容
        # patches = F.unfold(cv, kernel_size=patch_size, stride=stride)
        # # 将裁剪后的块融在batch维度进行处理
        # patches = patches.view(C, patch_size, patch_size, -1).permute(3, 0, 1, 2)  # [64, 28, 64, 64]
        # processed_patches = self.disparitygression(self.relu3(self.conv4(self.conv7(self.conv6(self.conv3(self.conv2(self.conv1(patches))))))))
        # # 将经过处理的滑动窗口数据重新折叠回原始形状的张量
        # output = processed_patches[:, :, pad_size:pad_size + stride, pad_size:pad_size + stride]  # 只选取 128*128 的中心大小为64*64作为输出
        # disp_raw = F.fold(output.permute(1, 2, 3, 0).reshape(1, -1, patches.size(0)),
        #                 output_size=(H, W), kernel_size=64, stride=64)
        
        
        
        # C, H, W = cv.shape[-3:]  # 获取输入张量的高度和宽度
        # # patch_size=64
        # # stride=56
        # # 对输入张量进行滑动窗口操作，将输入张量[N,C,H,W]展开成一个二维矩阵，每一行对应一个窗口的内容
        # patches = F.unfold(cv, kernel_size=patch_size, stride=stride)
        # # 将裁剪后的块融在batch维度进行处理
        # patches = patches.view(C, patch_size, patch_size, -1).permute(3, 0, 1, 2)  # [64, 28, 64, 64]
        # processed_patches = self.disparitygression(self.relu3(self.conv4(self.conv7(self.conv6(self.conv3(self.conv2(self.conv1(patches))))))))
        # # 将经过处理的滑动窗口数据重新折叠回原始形状的张量
        # output = F.fold(processed_patches.permute(1, 2, 3, 0).view(1, -1, patches.size(0)),
        #                 output_size=(H, W), kernel_size=patch_size, stride=stride)
        # # 计算每个位置的重叠次数
        # ones = torch.ones_like(cv)
        # counts = F.fold(F.unfold(ones, kernel_size=patch_size, stride=stride),
        #                 output_size=(H, W), kernel_size=patch_size, stride=stride)
        # # 归一化融合结果
        # disp_raw = output / counts
        
        
        
#         C, H, W = cv.shape[-3:]  # 获取输入张量的高度和宽度
#         patch_size=48
#         stride=29
#         # 对输入张量进行滑动窗口操作，将输入张量[N,C,H,W]展开成一个二维矩阵，每一行对应一个窗口的内容
#         patches = F.unfold(cv, kernel_size=patch_size, stride=stride)
#         # 将裁剪后的块融在batch维度进行处理
#         patches = patches.view(C, patch_size, patch_size, -1).permute(3, 0, 1, 2)  # [289, 28, 48, 48]
        
#         patches = patches * 255
#         patches = patches.to(torch.uint8)
#         np.save("input_crop_batch.npy", patches.permute(0, 2, 3, 1))
#         patches = patches / 255
        
#         processed_patches = self.disparitygression(self.relu3(self.conv4(self.conv7(self.conv6(self.conv3(self.conv2(self.conv1(patches))))))))
#         # 将经过处理的滑动窗口数据重新折叠回原始形状的张量
#         output = F.fold(processed_patches.permute(1, 2, 3, 0).view(1, -1, patches.size(0)),
#                         output_size=(H, W), kernel_size=patch_size, stride=stride)
#         # 计算每个位置的重叠次数
#         ones = torch.ones_like(cv)
#         counts = F.fold(F.unfold(ones, kernel_size=patch_size, stride=stride),
#                         output_size=(H, W), kernel_size=patch_size, stride=stride)
        
#         # patches = patches * 255
#         # patches = patches.to(torch.uint8)
#         # np.save("input_crop_batch.npy", patches)
#         # patches = patches / 255
        
#         # 归一化融合结果
#         disp_raw = output / counts
        
        
        
#         patches = []

#         for i in range(0, H, 64):
#             for j in range(0, W, 64):
#                 patch = cv[:, :, i:i+patch_size, j:j+patch_size]
#                 patches.append(patch)
#         processed_patches = []
#         for patch in patches:
#             cv = self.conv1(patch)
#             cv = self.conv2(cv)
#             cv = self.conv3(cv)
#             cv = self.conv6(cv)
#             cv = self.conv7(cv)
#             cv = self.conv4(cv)  # 注意将conv4的输出通道数设置为9
#             prob = self.relu3(cv)
#             disp_raw_crop = self.disparitygression(prob)
#             processed_patches.append(disp_raw_crop)
#         num_patches = (H // patch_size) * (W // patch_size)
#         disp_raw = torch.zeros(1, 1, H, W)
#         idx = 0
#         for i in range(0, H, patch_size):
#             for j in range(0, W, patch_size):
#                 disp_raw[:, :, i:i+patch_size, j:j+patch_size] = processed_patches[idx]
#                 idx += 1
#         print(disp_raw.shape)
   
        cv = self.conv1(cv)
        # cv = self.conv2(cv)
        # cv = self.conv3(cv)
        # cv = self.init_feature(cv)
        cv = self.resual_Block(cv)
        cv = self.resual_Block1(cv)
        # cv = self.conv6(cv)
        cv = self.conv7(cv)
        cv = self.conv4(cv)  # 注意将conv4的输出通道数设置为9
        # cv = self.relu(cv)
        
        # end_time2 = time.time()
        prob = self.relu3(cv)
        disp_raw = self.disparitygression(prob)
        # disp_final = self.finetune(disp_raw, x[3])
        raw_warp_img = []
        
        # end_time3 = time.time()
        # inference_time1 = (end_time1 - start_time) * 1000
        # inference_time2 = (end_time2 - end_time1) * 1000
        # inference_time3 = (end_time3 - end_time2) * 1000  # 转换为毫秒
        # print(f"cat time: {inference_time1:.2f} ms")
        # print(f"conv time: {inference_time2:.2f} ms")
        # print(f"softmaxregression time: {inference_time3:.2f} ms")

        return disp_raw, raw_warp_img

    
    def shift_49(self, disp, views_list):
        B,C,H,W = views_list[0].shape
        x, y = torch.arange(0, H), torch.arange(0, W)
        self.meshgrid = torch.stack(torch.meshgrid(x,y), -1).unsqueeze(0) # 1*H*W*2
        disp = disp.squeeze(1)
        # assert H==self.patch_size and W==self.patch_size,"size is different!"
        tmp = []
        meshgrid = self.meshgrid.repeat(B, 1, 1, 1).to(disp) # B*H*W*2
        for k in range(49):
            u, v = divmod(k, 7)
            grid = torch.stack([ 
                torch.clip(meshgrid[:,:,:,1]-disp*(v-3),0,W-1),
                torch.clip(meshgrid[:,:,:,0]-disp*(u-3),0,H-1)
            ],-1)/(W-1) *2 -1  # B*H*W*2  归一化到-1，1
            tmp.append(grid_sample(views_list[k], grid, align_corners=True))   
        return tmp
    
    def shift_81(self, disp, views_list):
        B,C,H,W = views_list[0].shape
        x, y = torch.arange(0, H), torch.arange(0, W)
        self.meshgrid = torch.stack(torch.meshgrid(x,y), -1).unsqueeze(0) # 1*H*W*2
        disp = disp.squeeze(1)
        # assert H==self.patch_size and W==self.patch_size,"size is different!"
        tmp = []
        meshgrid = self.meshgrid.repeat(B, 1, 1, 1).to(disp) # B*H*W*2
        for k in range(81):
            u, v = divmod(k, 9)
            grid = torch.stack([ 
                torch.clip(meshgrid[:,:,:,1]-disp*(v-4),0,W-1),
                torch.clip(meshgrid[:,:,:,0]-disp*(u-4),0,H-1)
            ],-1)/(W-1) *2 -1  # B*H*W*2  归一化到-1，1
            tmp.append(grid_sample(views_list[k], grid, align_corners=True))   
        return tmp
    
    
    def shift(self, disp, views_list):
        B,C,H,W = views_list[0].shape
        x, y = torch.arange(0, H), torch.arange(0, W)
        self.meshgrid = torch.stack(torch.meshgrid(x,y), -1).unsqueeze(0) # 1*H*W*2
        disp = disp.squeeze(1)
        # assert H==self.patch_size and W==self.patch_size,"size is different!"
        tmp = []
        meshgrid = self.meshgrid.repeat(B, 1, 1, 1).to(disp) # B*H*W*2
        for k in range(36):
            u, v = divmod(index[k], 9)
            grid = torch.stack([ 
                torch.clip(meshgrid[:,:,:,1]-disp*(v-4),0,W-1),
                torch.clip(meshgrid[:,:,:,0]-disp*(u-4),0,H-1)
            ],-1)/(W-1) *2 -1  # B*H*W*2  归一化到-1，1
            tmp.append(grid_sample(views_list[k], grid, align_corners=True))   
        return tmp
    
    def warp(self, disp, views_list, idx):
        B,C,H,W = views_list[0].shape
        x, y = torch.arange(0, H), torch.arange(0, W)
        self.meshgrid = torch.stack(torch.meshgrid(x,y), -1).unsqueeze(0) # 1*H*W*2
        disp = disp.squeeze(1)
        # assert H==self.patch_size and W==self.patch_size,"size is different!"
        tmp = []
        meshgrid = self.meshgrid.repeat(B, 1, 1, 1).to(disp) # B*H*W*2
        for k in range(9):
            u, v = divmod(idx[k], 9)
            grid = torch.stack([ 
                torch.clip(meshgrid[:,:,:,1]-disp*(v-4),0,W-1),
                torch.clip(meshgrid[:,:,:,0]-disp*(u-4),0,H-1)
            ],-1)/(W-1) *2 -1  # B*H*W*2  归一化到-1，1
            tmp.append(grid_sample(views_list[k], grid, align_corners=True))   
        return tmp

    def disparitygression(self, input):
        disparity_values = torch.linspace(-4,4,9,device=input.device)
        x = disparity_values.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        out = torch.sum(torch.multiply(input, x), 1)
        return out.unsqueeze(1)

    def cal_occlusion(self, views):
        views = torch.stack(views, -1) # B*C*H*W*9
        if self.grad_v == 9:
            grad = torch.mean(torch.abs(views[:,:,:,:,1:]- views[:,:,:,:,:8]), dim=[1,4])
        elif self.grad_v == 5:
            grad = torch.mean(torch.abs(views[:,:,:,:,3:7] - views[:,:,:,:,2:6]), dim=[1,4]) # B*H*W   
        return grad.unsqueeze(1)
    