import torch
from torch.nn.functional import grid_sample, adaptive_avg_pool2d
from torch import nn, Tensor
from BinOp import BinOp
from .loss import Lossv3, Lossv9, Lossv5, get_smooth_loss, noOPAL, Lossv4
from .basemodel import BaseModel, init_net, get_optimizer
from .layers import *
import time
from brevitas.nn import QuantConv2d,QuantReLU
from brevitas.quant.scaled_int import Int8ActPerTensorFloat
import torch.nn.functional as F
import numpy as np
from brevitas.quant.scaled_int import Int8ActPerTensorFloat,Int8WeightPerChannelFloat,Uint8ActPerTensorFloatMaxInit
from brevitas.quant import SignedBinaryWeightPerTensorConst
from brevitas.core.quant import BinaryQuant,ClampedBinaryQuant,QuantType
from brevitas.core.scaling import ConstScaling,ParameterScaling,ParameterFromStatsFromParameterScaling
from brevitas.inject import ExtendedInjector,value,this
from brevitas.proxy import WeightQuantProxyFromInjector,ActQuantProxyFromInjector
from brevitas.nn import QuantConv2d,QuantReLU,QuantIdentity
from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.restrict_val import FloatToIntImplType,RestrictValueType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.quant.solver import ActQuantSolver,WeightQuantSolver

from brevitas.inject.enum import *

from brevitas.inject import ExtendedInjector
from brevitas.proxy import ActQuantProxyFromInjector

from brevitas.inject import value
from brevitas.proxy import WeightQuantProxyFromInjector
from brevitas.core.scaling import ParameterScaling
from brevitas.core.quant import IntQuant, RescalingIntQuant


from brevitas.core.quant import ClampedBinaryQuant
from brevitas.proxy import WeightQuantProxyFromInjector, ActQuantProxyFromInjector
from brevitas.inject import this
from brevitas.core.function_wrapper import TensorClamp, OverTensorView, RoundSte
from brevitas.core.scaling import ParameterFromRuntimeStatsScaling, IntScaling
from brevitas.core.stats import AbsPercentile
from brevitas.core.restrict_val import FloatRestrictValue
from brevitas.core.bit_width import BitWidthConst
from brevitas.core.quant import IntQuant, RescalingIntQuant
from brevitas.core.zero_point import ZeroZeroPoint

from brevitas.core.function_wrapper import InplaceTensorClampSte
from brevitas.core.function_wrapper import TensorClamp
from brevitas.quant.base import SignedBinaryClampedConst
from brevitas.quant.solver import ActQuantSolver
from brevitas.quant.solver import WeightQuantSolver
from brevitas.core.restrict_val import RestrictValueType
from dependencies import value

from brevitas.inject import ExtendedInjector
from brevitas.quant.solver import WeightQuantSolver, ActQuantSolver
from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import RestrictValueType, FloatToIntImplType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.quant import Uint8ActPerTensorFloatMaxInit, Int8ActPerTensorFloatMinMaxInit
from brevitas.quant import Int8WeightPerTensorFloat
import torch
from torch.nn.functional import grid_sample, adaptive_avg_pool2d
from torch import nn, Tensor
import time
from brevitas.nn import QuantConv2d,QuantReLU
from brevitas.quant.scaled_int import Int8ActPerTensorFloat
import torch.nn.functional as F
from brevitas.quant.scaled_int import Int8ActPerTensorFloat,Int8WeightPerChannelFloat,Uint8ActPerTensorFloatMaxInit
from brevitas.quant import SignedBinaryWeightPerTensorConst
from brevitas.core.quant import BinaryQuant,ClampedBinaryQuant,QuantType
from brevitas.core.scaling import ConstScaling,ParameterScaling,ParameterFromStatsFromParameterScaling
from brevitas.inject import ExtendedInjector,value,this
from brevitas.proxy import WeightQuantProxyFromInjector,ActQuantProxyFromInjector
from brevitas.nn import QuantConv2d,QuantReLU,QuantIdentity
from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.restrict_val import FloatToIntImplType,RestrictValueType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.quant.solver import ActQuantSolver,WeightQuantSolver

from brevitas.inject.enum import *

from brevitas.inject import ExtendedInjector
from brevitas.proxy import ActQuantProxyFromInjector

from brevitas.inject import value
from brevitas.proxy import WeightQuantProxyFromInjector
from brevitas.core.scaling import ParameterScaling
from brevitas.core.quant import IntQuant, RescalingIntQuant


from brevitas.core.quant import ClampedBinaryQuant
from brevitas.proxy import WeightQuantProxyFromInjector, ActQuantProxyFromInjector
from brevitas.inject import this
from brevitas.core.function_wrapper import TensorClamp, OverTensorView, RoundSte
from brevitas.core.scaling import ParameterFromRuntimeStatsScaling, IntScaling
from brevitas.core.stats import AbsPercentile
from brevitas.core.restrict_val import FloatRestrictValue
from brevitas.core.bit_width import BitWidthConst
from brevitas.core.quant import IntQuant, RescalingIntQuant
from brevitas.core.zero_point import ZeroZeroPoint

from brevitas.core.function_wrapper import InplaceTensorClampSte
from brevitas.core.function_wrapper import TensorClamp
from brevitas.quant.base import SignedBinaryClampedConst
from brevitas.quant.solver import ActQuantSolver
from brevitas.quant.solver import WeightQuantSolver
from brevitas.core.restrict_val import RestrictValueType


import torch
from torch.nn import BatchNorm1d
from torch.nn import BatchNorm2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList

from brevitas.core.restrict_val import RestrictValueType
from brevitas.nn import QuantConv2d
from brevitas.nn import QuantIdentity
from brevitas.nn import QuantLinear
from dependencies import value

from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import FloatToIntImplType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.inject import ExtendedInjector
from brevitas.quant.solver import ActQuantSolver
from brevitas.quant.solver import WeightQuantSolver


from typing import List

from torch import Tensor
import torch.nn as nn

import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerChannelFloat
from brevitas.quant import Int8WeightPerTensorFloat,Uint8ActPerTensorFloat
from brevitas.quant import IntBias
from brevitas.quant import TruncTo8bit
from brevitas.quant_tensor import QuantTensor

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
    scaling_const = 1.0


class CommonActQuant(CommonQuant, ActQuantSolver):
    min_val = -1.0
    max_val = 1.0


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
    
IDX = [[36 + j for j in range(9)],
       [4 + 9 * j for j in range(9)],
       [10 * j for j in range(9)],
       [8 * (j + 1) for j in range(9)]]

# 没有残差的特征提取模块
class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,  downsample=None):
        super(conv_block, self).__init__()

        block = nn.ModuleList()
        block.append(BuildingBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, dilation=1, padding=1, 
                          groups=1, bias=False, weight_bit_width=4, act_bit_width=4))
        block.append(BuildingBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, dilation=1, padding=1, 
                          groups=1, bias=False, weight_bit_width=4, act_bit_width=4))
        self.conv = nn.Sequential(*block)

    def forward(self, x):

        return self.conv(x)

class Feature(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Feature, self).__init__()
        pad = kernel_size // 2
        self.relu = nn.ReLU(True)
        self.conv1 = BuildingBlock(in_channels=3, out_channels=4, kernel_size=3, stride=1, dilation=1, padding=1, 
                          groups=1, bias=False, weight_bit_width=8, act_bit_width=4)

        self.layer1 = self._make_layer(4, 4, 2, 1)
        self.layer2 = self._make_layer(4, 8, 2, 1)
        self.layer3 = self._make_layer(8, 16, 2, 1)

        self.branch = BuildingBlock(in_channels=16, out_channels=28, kernel_size=1, stride=1, dilation=1, padding=0, 
                          groups=1, bias=False, weight_bit_width=4, act_bit_width=4)
        self.lastconv = nn.Sequential(BuildingBlock(in_channels=28, out_channels=16, kernel_size=3, stride=1, dilation=1, padding=1, 
                          groups=1, bias=False, weight_bit_width=4, act_bit_width=4), BuildingBlock(in_channels=16, out_channels=8, kernel_size=1, stride=1, dilation=1, padding=0, 
                          groups=1, bias=False, weight_bit_width=4, act_bit_width=4))

    def _make_layer(self, in_c, out_c, blocks, stride):
        layers = []
        layers.append(conv_block(in_c, out_c, 3, stride))
        for _ in range(1, blocks):
            layers.append(conv_block(out_c, out_c, 3, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        x = self.branch(l3)
        x = self.lastconv(x)
        return x

    
    
    
class ContextAdjustmentLayer(nn.Module):

    def __init__(self, num_blocks=8, feature_dim=16, expansion=3):
        super().__init__()
        self.num_blocks = num_blocks
        self.in_conv = BinConv2d(4, feature_dim, kernel_size=3, padding=1)
        self.layers = nn.ModuleList([ResBlock(feature_dim, expansion) for _ in range(num_blocks)])
        self.out_conv = QuantConv2d(feature_dim, 1, kernel_size=3, padding=1,bias=False,input_quant=Int8ActPerTensorFloat)

    def forward(self, disp_raw: Tensor, img: Tensor):
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
            BinConv2d(n_feats + 1, n_feats * expansion_ratio, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            BinConv2d(n_feats * expansion_ratio, n_feats, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor, disp: torch.Tensor):
        return x + self.module(torch.cat([disp, x], dim=1)) * self.res_scale

# 将权重和激活均进行1bit量化，当使用scaling_const时,效果最好的MSE=5.942，但一般情况下MSE处于9~10之间,当使用scaling_init时,MSE=9.626
class CommonQuant(ExtendedInjector): # 这段代码定义了一个 CommonQuant 类，它继承自 ExtendedInjector。这个类包含了几个类属性和一个方法，用于处理量化（quantization）的相关设置。
    bit_width_impl_type = BitWidthImplType.CONST # 量化中的位宽实现类型为常量
    scaling_impl_type = ScalingImplType.CONST # 量化中的缩放因子实现类型为常量。这意味着在量化过程中，缩放因子的值是固定的，而不是根据数据动态调整的。
    # scaling_impl_type = ScalingImplType.STATS # scale based on statistics
    # scaling_stats_op = StatsOp.MAX # scale statistics is the absmax value
    restrict_scaling_type = RestrictValueType.FP # 表示缩放因子的限制类型为浮点数。这可能意味着量化过程中使用的缩放因子以浮点数形式表示，而不是整数。
    zero_point_impl = ZeroZeroPoint # 在量化过程中零点的实现方式是固定的 ZeroZeroPoint 类型
    float_to_int_impl_type = FloatToIntImplType.ROUND # 将浮点数转换为整数时使用四舍五入的方法
    scaling_per_output_channel = False # 在量化过程中，不会为每个输出通道单独设置缩放因子，而是使用统一的缩放因子。
    narrow_range = True # 在量化过程中使用的数值范围较窄，可能会影响模型的精度和范围。
    signed = True # 量化过程中使用的整数表示是有符号的，即可以表示负数和正数。

    @value
    def quant_type(bit_width): # 根据位宽 (bit_width) 确定量化类型 (QuantType)
        if bit_width is None:
            return QuantType.FP
        elif bit_width == 1:
            return QuantType.BINARY
        else:
            return QuantType.INT


class CommonWeightQuant(CommonQuant, WeightQuantSolver): # CommonWeightQuant 类继承自 CommonQuant 和 WeightQuantSolver，专注于权重量化，设置了一个固定的缩放常数。
    # scaling_per_output_channel = True
    # @value
    # def scaling_const(module):
    #     negMean = module.weight.mean(1, keepdim=True).mul(-1).expand_as(module.weight)
    #     num_pixel = module.weight[0].nelement()
    #     a = module.weight.add(negMean).clamp(-1.0, 1.0).norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(num_pixel)
    #     #a = module.weight.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(num_pixel)
    #     #print(a[0:3])
    #     return a
    scaling_const = 1.0


class CommonActQuant(CommonQuant, ActQuantSolver): # CommonActQuant 类继承自 CommonQuant 和 ActQuantSolver，用于激活量化，并定义了激活值的最小和最大范围。
    #scaling_const = 1.0
    min_val = -1.0
    max_val = 1.0    


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
    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output) 
        return output
    
    
    
class BinConv2d(nn.Module):
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=1, padding=0, groups=1, dropout=0,
            Linear=False, previous_conv=False, size=0):
        super(BinConv2d, self).__init__()
        self.input_channels = input_channels
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout
        self.previous_conv = previous_conv
        
        self.conv = QuantConv2d(
                    input_channels, 
                    output_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=4)
        self.bn = nn.BatchNorm2d(output_channels, eps=1e-4, momentum=0.1, affine=True)
        #self.relu = QuantIdentity(act_quant=CommonActQuant, bit_width=1)
        #self.relu = QuantReLU(act_quant=CommonUintActQuant,bit_width=1)
        self.relu = QuantReLU(act_quant=CommonUintActQuant,bit_width=4)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class OPENetmodel(BaseModel):  # Basemodel
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['L1']
        if opt.losses.find(
                'smooth') != -1:  # 判断 opt.losses 字符串中是否包含子字符串 'smooth'，如果包含则向 self.loss_names 列表中添加一个元素 'smoothness'。
            self.loss_names.append('smoothness')
        self.visual_names = ['center_input', 'output', 'label']

        self.model_names = ['EPI']
        net = eval(self.opt.net_version)(opt, self.device,
                                         self.isTrain)  # eval(self.opt.net_version)的作用是返回网络版本，即下面的Unsup31_15_53、Unsup27_16_16或Unsup26_21_56

        self.netEPI = init_net(net, opt.init_type, opt.init_gain, self.gpu_ids)
        self.use_v = opt.use_views  # 表示9*9的子孔径图像，索引为0~8（一行或者一列），use_views为9

        self.center_index = self.use_v // 2  # 将use_views的一半作为中心视图的索引：4
        self.alpha = opt.alpha
        self.pad = opt.pad
        self.lamda = opt.lamda

        self.test_loss_log = 0  # 将初始测试损失值设为 0。
        self.test_loss = torch.nn.L1Loss()  # 创建一个 L1 损失函数对象
        if self.isTrain:
            # define loss functions
            self.criterionL1 = eval(self.opt.loss_version)(self.opt, self.device)
            self.optimizer = get_optimizer(opt, filter(lambda p: p.requires_grad, self.netEPI.parameters()), LR=opt.lr)
            self.optimizers.append(self.optimizer)

    def set_input(self, inputs, epoch):
        self.epoch = epoch
        # self.supervise_view = rearrange(inputs[0].to(self.device), 'b c (h1 h) (w1 w) u v -> (b h1 w1) c h w u v', h1=8, w1=8)
        self.supervise_view = inputs[0].to(
            self.device)  # 这段代码将输入数据 inputs[0] 移动（或复制）到指定的设备上，即将数据传输到 GPU 或 CPU 上进行处理。具体来说，inputs[0] 是模型的输入数据，通过调用 .to(self.device) 方法，将其移动到在类中定义的 self.device 上。
        # 移动数据到特定设备上的操作通常用于确保模型在训练或推理过程中能够利用 GPU 等加速器来加快计算速度。通过在不同设备上存储和处理数据，可以充分利用硬件资源，提高模型的性能表现。
        self.input = []
        for j in range(self.use_v): 
            self.input.append(self.supervise_view[:, :, :, :, self.center_index, j])  # 行
        for j in range(self.use_v):
            self.input.append(self.supervise_view[:, :, :, :, j, self.center_index])  # 列
        for j in range(self.use_v):
            self.input.append(self.supervise_view[:, :, :, :, j, j])  # 对角线
        for j in range(self.use_v):
            self.input.append(self.supervise_view[:, :, :, :, j, self.use_v - 1 - j])
        # ------------输入是来自四个不同方向的子孔径图像---------------
        self.center_input = self.input[self.center_index]
        self.label = inputs[1].to(self.device)

    # self.netEPI(self.input) 表示将输入数据 self.input 传递给网络模型 self.netEPI 进行计算，得到的结果赋值给 self.output 和 self.raw_warp_img。
    # self.output 是模型的输出结果，self.raw_warp_img 是网络模型中间的原始扭曲图像。
    def forward(self):  #############################################################################
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.output, self.raw_warp_img = self.netEPI(self.input)  # G(A)
        self.test_loss_log = 0

    def backward_G(self):
        # if self.epoch <= self.opt.n_epochs:
        #     self.loss_L1 = self.criterionL1(self.output, self.input[:9], self.input[9:18])
        # else:
        #     self.loss_L1 = self.criterionL1_5(self.output, self.input[:9], self.input[9:18])
        # self.loss_L1 = self.criterionL1(self.output, self.input[:9], self.input[9:18])
        self.loss_L1 = self.criterionL1(self.epoch, self.output, self.input[:9],
                                        self.input[9:18])  # OPENet-fast，self.input[:9], self.input[9:18]分别代表水平喝垂直方向
        # 通常 L1 损失是指 Mean Absolute Error（MAE），用于衡量模型输出与目标数据之间的绝对差异。
        # self.epoch 可能是当前训练的 epoch 数，用于在损失函数中引入一些动态调整或变化；self.output 是模型的输出结果；
        # 该行代码的作用是调用损失函数 self.criterionL1，并将相关参数传递给该函数，计算得到 L1 损失值。
        if self.raw_warp_img is None:
            self.loss_total = self.loss_L1  # 此处的raw_warp_img指什么
        else:
            self.loss_raw = 0
            for views in self.raw_warp_img:
                self.loss_raw += self.criterionL1(self.epoch, views)
            self.loss_total = 0.6 * self.loss_L1 + 0.4 * self.loss_raw  # 从此处可以看出loss_L1是论文中raw损失，loss_raw是final损失
            # self.loss_total = self.loss_L1
        self.loss_smoothness = get_smooth_loss(self.output, self.center_input, self.lamda)

        if 'smoothness' in self.loss_names and self.epoch > 2 * self.opt.n_epochs:
            self.loss_total += self.loss_smoothness
        self.loss_total.backward()


    def optimize_parameters(self):
        """
        首先定义二值化操作符
        """
        #bin_op = BinOp(self.netEPI)
        self.netEPI.train()
        '''
        模型开始运转之前先对权重进行二值化处理
        '''
        #bin_op.binarization()
        self.forward()
        self.optimizer.zero_grad()  # 将优化器中所有参数的梯度清零，以准备接收新一轮的梯度计算。
        self.backward_G()
        '''
        根据loss.backward()得到梯度之后先进行权重的恢复然后计算二值化权重的梯度
        '''
        #bin_op.restore()
        #bin_op.updateBinaryGradWeight()
        self.optimizer.step()  # 根据优化器中存储的参数梯度信息，更新模型的参数。

# resnet18_full
# from typing import List

# from torch import Tensor
# import torch.nn as nn

# import brevitas.nn as qnn
# from brevitas.quant import Int8WeightPerChannelFloat
# from brevitas.quant import Int8WeightPerTensorFloat
# from brevitas.quant import IntBias
# from brevitas.quant import TruncTo8bit
# from brevitas.quant_tensor import QuantTensor


# def make_quant_conv2d(
#         in_channels,
#         out_channels,
#         kernel_size,
#         weight_bit_width,
#         weight_quant,
#         stride=1,
#         padding=0,
#         bias=False):
#     return qnn.QuantConv2d(
#         in_channels=in_channels,
#         out_channels=out_channels,
#         kernel_size=kernel_size,
#         stride=stride,
#         padding=padding,
#         bias=bias,
#         weight_quant=weight_quant,
#         weight_bit_width=weight_bit_width)


# class QuantBasicBlock(nn.Module):
#     """
#     Quantized BasicBlock implementation with extra relu activations to respect FINN constraints on the sign of residual
#     adds. Ok to train from scratch, but doesn't lend itself to e.g. retrain from torchvision.
#     """
#     expansion = 1

#     def __init__(
#             self,
#             in_planes,
#             planes,
#             stride=1,
#             bias=False,
#             shared_quant_act=None,
#             act_bit_width=8,
#             weight_bit_width=8,
#             weight_quant=Int8WeightPerChannelFloat):
#         super(QuantBasicBlock, self).__init__()
#         self.conv1 = make_quant_conv2d(
#             in_planes,
#             planes,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             bias=bias,
#             weight_bit_width=weight_bit_width,
#             weight_quant=weight_quant)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu1 = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)
#         self.conv2 = make_quant_conv2d(
#             planes,
#             planes,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             bias=bias,
#             weight_bit_width=weight_bit_width,
#             weight_quant=weight_quant)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.downsample = nn.Sequential(
#                 make_quant_conv2d(
#                     in_planes,
#                     self.expansion * planes,
#                     kernel_size=1,
#                     stride=1,
#                     padding=0,
#                     bias=bias,
#                     weight_bit_width=weight_bit_width,
#                     weight_quant=weight_quant),
#                 nn.BatchNorm2d(self.expansion * planes),
#                 # We add a ReLU activation here because FINN requires the same sign along residual adds
#                 qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True))
#             # Redefine shared_quant_act whenever shortcut is performing downsampling
#             shared_quant_act = self.downsample[-1]
#         if shared_quant_act is None:
#             shared_quant_act = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)
#         # We add a ReLU activation here because FINN requires the same sign along residual adds
#         self.relu2 = shared_quant_act
#         self.relu_out = qnn.QuantReLU(return_quant_tensor=True, bit_width=act_bit_width)

#     def forward(self, x):
#         out = self.relu1(self.bn1(self.conv1(x)))
#         out = self.relu2(self.bn2(self.conv2(out)))
#         if len(self.downsample):
#             x = self.downsample(x)
#         # Check that the addition is made explicitly among QuantTensor structures
#         assert isinstance(out, QuantTensor), "Perform add among QuantTensors"
#         assert isinstance(x, QuantTensor), "Perform add among QuantTensors"
#         out = out + x
#         out = self.relu_out(out)
#         return out  
    
    
# class Unsup27_16_16(nn.Module):  # v3
#     def __init__(
#             self,
#             opt,
#             device,
#             is_train=True,
#             block_impl=QuantBasicBlock,
#             num_blocks=[2, 2, 2, 2],
#             zero_init_residual=False,
#             act_bit_width=4,
#             weight_bit_width=4,
#             last_layer_bias_quant=IntBias,
#             weight_quant=Int8WeightPerChannelFloat,
#             first_layer_weight_quant=Int8WeightPerChannelFloat,
#             last_layer_weight_quant=Int8WeightPerTensorFloat):
#         super().__init__()
#         self.is_train = is_train  # 设置是否为训练模式，默认为 True。
#         self.n_angle = 2  # 角度数量为 2，可能与后续涉及视角相关的操作有关。
#         self.device = device
#         self.use_v = opt.use_views  # 使用的视角数，根据外部传入的参数确定。
#         self.grad_v = opt.grad_v  # 梯度计算使用的视角数，同样根据外部参数确定。
#         self.in_planes = 64
#         self.conv1 = make_quant_conv2d(
#             3,
#             64,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             weight_bit_width=8,
#             weight_quant=first_layer_weight_quant)
#         self.bn1 = nn.BatchNorm2d(64)
#         shared_quant_act = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)
#         self.relu = shared_quant_act

#         self.layer1, shared_quant_act = self._make_layer(
#             block_impl, 64, num_blocks[0], 1, shared_quant_act, weight_bit_width, act_bit_width, weight_quant)
#         self.layer2, shared_quant_act = self._make_layer(
#             block_impl, 128, num_blocks[1], 2, shared_quant_act, weight_bit_width, act_bit_width, weight_quant)
#         self.layer3, shared_quant_act = self._make_layer(
#             block_impl, 256, num_blocks[2], 2, shared_quant_act, weight_bit_width, act_bit_width, weight_quant)
#         self.lastconv = make_quant_conv2d(
#             256,
#             9,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             bias=False,
#             weight_bit_width=8,
#             weight_quant=CommonWeightQuant)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def _make_layer(
#             self,
#             block_impl,
#             planes,
#             num_blocks,
#             stride,
#             shared_quant_act,
#             weight_bit_width,
#             act_bit_width,
#             weight_quant):
#         strides = [stride] + [1] * (num_blocks - 1)  # 1 在括号里是为了生成一个列表，该列表包含 (num_blocks - 1) 个 1。使用 + 操作符将 [stride] 和 [1] * (num_blocks - 1) 合并成一个新的列表。
#         layers = []
#         for stride in strides:
#             block = block_impl(
#                 in_planes=self.in_planes,
#                 planes=planes,
#                 stride=stride,
#                 bias=False,
#                 shared_quant_act=shared_quant_act,
#                 act_bit_width=act_bit_width,
#                 weight_bit_width=weight_bit_width,
#                 weight_quant=weight_quant)
#             layers.append(block)
#             shared_quant_act = layers[-1].relu_out
#             self.in_planes = planes * block_impl.expansion
#         return nn.Sequential(*layers), shared_quant_act


#     def forward(self, x):   # 输入是来自四个不同方向的子孔径图像
#         # There is no input quantizer, we assume the input is already 8b RGB
#         out = self.relu(self.bn1(self.conv1(x[0])))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.lastconv(out)

#         disp_raw = self.disparitygression(out)
#         occolusion = []
#         disp = []   # 视差图
#         mask = []
#         raw_warp_img = []   # 初始的变换视图
#         for j in range(self.n_angle):   # 这段代码是一个 for 循环，用于处理多个角度的数据。在每次循环中，根据当前角度 j，进行一系列操作，不过此处只取了水平方向和垂直方向的子孔径图像以及相对应的索引
#             warpped_views = self.warp(disp_raw, x[self.use_v * j:self.use_v * (j + 1)], IDX[j])
#             if self.is_train:
#                 raw_warp_img.append(warpped_views)
#         #     disp_final = self.finetune(disp_raw, x[4])
#         #     disp.append(disp_final)
#         # disp = torch.cat(disp, 1)
#         # disp_mean = torch.mean(disp, 1, True)
        
#         return disp_raw, raw_warp_img
class Residual(nn.Module):
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=1, padding=0, groups=1, dropout=0,
            Linear=False, previous_conv=False, size=0):
        super(Residual, self).__init__()
        self.input_channels = input_channels
        self.layer_type = 'Residual'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout
        self.previous_conv = previous_conv
        
        self.conv1 = BinConv2d(input_channels, 3, 3, 1, padding=1)
        self.fuse2d = nn.ModuleList()
        for j in range(4):
            self.fuse2d.append(nn.Sequential(BinConv2d(3, 3, 3, 1, padding=1),BinConv2d(3, 3, 1, 1)))
        self.quant = QuantReLU(act_quant=CommonUintActQuant,bit_width=4)
        self.fuse3 = BinConv2d(3, output_channels, 3, 1, padding=1)

    def forward(self, x):
        cv = self.conv1(x)
        for j in range(4):
            cv = self.quant(self.fuse2d[j](cv)) + self.quant(cv)
        cv = self.fuse3(cv)  # 经过fuse3以及relu3操作后得到了概率
        
        return cv
# class Residual(nn.Module):
#     def __init__(self, input_channels, output_channels,
#             kernel_size=-1, stride=1, padding=0, groups=1, dropout=0,
#             Linear=False, previous_conv=False, size=0):
#         super(Residual, self).__init__()
#         self.input_channels = input_channels
#         self.layer_type = 'Residual'
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dropout_ratio = dropout
#         self.previous_conv = previous_conv
        
#         self.conv1 = BinConv2d(input_channels, 3, 3, 1, padding=1)
#         self.fuse2d = nn.ModuleList()
#         for j in range(4):
#             self.fuse2d.append(nn.Sequential(BinConv2d(3, 3, 3, 1, padding=1),BinConv2d(3, 3, 1, 1)))
#         self.quant = QuantReLU(act_quant=CommonUintActQuant,bit_width=4)
#         self.fuse3 = BinConv2d(3, output_channels, 3, 1, padding=1)

#     def forward(self, x):
#         cv = self.conv1(x)
#         for j in range(4):
#             cv = self.quant(self.fuse2d[j](cv)) + self.quant(cv)
#         cv = self.fuse3(cv)  # 经过fuse3以及relu3操作后得到了概率
        
#         return cv

class Unsup27_16_16(nn.Module):  # v3
    def __init__(self, opt, device, is_train=True):
        super().__init__()
        self.is_train = is_train  # 设置是否为训练模式，默认为 True。
        self.n_angle = 2  # 角度数量为 2，可能与后续涉及视角相关的操作有关。
        feats = 64  # 特征通道数为 64，后续会根据需要调整。
        self.device = device
        self.use_v = opt.use_views  # 使用的视角数，根据外部传入的参数确定。
        self.grad_v = opt.grad_v  # 梯度计算使用的视角数，同样根据外部参数确定。
        # self.feat_extract = Feature(in_channels=opt.input_c, out_channels=8)
        # self.block3d = nn.ModuleList()
        # self.fuse3d = nn.ModuleList()
        # for _ in range(4):
        #     self.block3d.append(
        #         nn.Sequential(BinConv2d(72, 64, 3, 1, 1), nn.ReLU(True)))
        #     self.fuse3d.append(nn.Sequential(BinConv2d(64, 32, 1, 1), nn.ReLU(True)))
        feats *= 2
        # self.fuse2d = nn.ModuleList()
        # for j in range(4):
        #     self.fuse2d.append(nn.Sequential(BinConv2d(feats, feats, 3, 1, padding=1), nn.ReLU(True),
        #                                      BinConv2d(feats, feats, 1, 1)))
        #     self.fuse2d.append(nn.ReLU(True))
        # self.fuse3 = BinConv2d(feats, 9, 3, 1, padding=1)
        # self.relu3 = nn.Softmax(dim=1)
        # self.finetune = ContextAdjustmentLayer()
        
        
        # opal_bnn
        self.feat_extract = Feature(in_channels=3, out_channels=8)
        # self.block3d = nn.ModuleList()
        # self.fuse3d = nn.ModuleList()
        # for _ in range(4):
        #     self.block3d.append(BinConv2d(27, 3, 3, 1, 1))
        #     self.fuse3d.append(BinConv2d(3, 3, 1, 1))
        self.feat_fuse = nn.Sequential(BuildingBlock(in_channels=72, out_channels=64, kernel_size=3, stride=1, dilation=1, padding=1, 
                          groups=1, bias=False, weight_bit_width=4, act_bit_width=4),BuildingBlock(in_channels=64, out_channels=32, kernel_size=1, stride=1, dilation=1, padding=0, 
                          groups=1, bias=False, weight_bit_width=4, act_bit_width=4))
        # self.feat_residual = Residual(12,9)
        self.feat_residual = nn.Sequential(BuildingBlock(in_channels=feats, out_channels=feats, kernel_size=3, stride=1, dilation=1, padding=1, 
                          groups=1, bias=False, weight_bit_width=4, act_bit_width=4),BuildingBlock(in_channels=feats, out_channels=feats, kernel_size=1, stride=1, dilation=1, padding=0, 
                          groups=1, bias=False, weight_bit_width=4, act_bit_width=4),BuildingBlock(in_channels=feats, out_channels=feats, kernel_size=3, stride=1, dilation=1, padding=1, 
                          groups=1, bias=False, weight_bit_width=4, act_bit_width=4),BuildingBlock(in_channels=feats, out_channels=feats, kernel_size=1, stride=1, dilation=1, padding=0, 
                          groups=1, bias=False, weight_bit_width=4, act_bit_width=4),BuildingBlock(in_channels=feats, out_channels=feats, kernel_size=3, stride=1, dilation=1, padding=1, 
                          groups=1, bias=False, weight_bit_width=4, act_bit_width=4),BuildingBlock(in_channels=feats, out_channels=feats, kernel_size=1, stride=1, dilation=1, padding=0, 
                          groups=1, bias=False, weight_bit_width=4, act_bit_width=4),BuildingBlock(in_channels=feats, out_channels=feats, kernel_size=3, stride=1, dilation=1, padding=1, 
                          groups=1, bias=False, weight_bit_width=4, act_bit_width=4),BuildingBlock(in_channels=feats, out_channels=feats, kernel_size=1, stride=1, dilation=1, padding=0, 
                          groups=1, bias=False, weight_bit_width=4, act_bit_width=4),QuantConv2d(in_channels=feats, out_channels=9, kernel_size=3, stride=1, padding=1,bias=False, 
                            weight_quant=CommonIntWeightPerChannelQuant, weight_bit_width=8))
        self.relu3 = nn.Softmax(dim=1)
        
        
        # concat
        # self.conv1 = BinConv2d(108, 3, 3, 1, padding=1)
        # self.fuse2d = nn.ModuleList()
        # for j in range(4):
        #     self.fuse2d.append(nn.Sequential(BinConv2d(3, 3, 3, 1, padding=1),BinConv2d(3, 3, 1, 1)))
        # self.quant = QuantReLU()
        # self.fuse3 = BinConv2d(3, 9, 3, 1, padding=1)
        # self.relu3 = nn.Softmax(dim=1)
        
        
        # residual
        # self.conv1 = BinConv2d(3, 3, 3, 1, padding=1)
        # self.fuse2d = nn.ModuleList()
        # for j in range(4):
        #     self.fuse2d.append(nn.Sequential(BinConv2d(3, 3, 3, 1, padding=1),BinConv2d(3, 3, 1, 1)))
        #     #self.fuse2d.append(QuantReLU(act_quant=CommonUintActQuant,bit_width=2))
        #     #self.fuse2d.append(QuantReLU())
        # self.quant = QuantReLU()
        # self.fuse3 = BinConv2d(3, 9, 3, 1, padding=1)
        # self.relu3 = nn.Softmax(dim=1)

        
        # resnet_add
        # self.conv1 = BinConv2d(3, 3, 3, 1, padding=1)
        # self.fuse2d = nn.ModuleList()
        # self.quant1 = QuantReLU()
        # self.quant2 = QuantReLU()
        # self.quant3 = QuantReLU()
        # self.quant4 = QuantReLU()
        # for j in range(4):
        #     self.fuse2d.append(nn.Sequential(BinConv2d(3, 3, 3, 1, padding=1),BinConv2d(3, 3, 1, 1)))
        #     self.fuse2d.append(QuantReLU())
        # self.fuse3 = BinConv2d(3, 9, 3, 1, padding=1)


        
    def forward(self, x):
        # 加载测试数据并从512*512的图片中随机裁剪出64*64的图片
        # 注意在加载测试数据的时候将/255去掉,将batchsize改为1
        image = torch.cat(x[:], 0)
        np.save("input1.npy", image.cpu())
        print(image.dtype)
        image = image.type(torch.uint8)
        print(image.dtype)
        np.save("input2.npy", image.cpu())
        print(image.shape)
        crop_size = 64
        _, _, h, w = image.shape
        # 确保图像尺寸足够大以进行裁剪
        if h < crop_size or w < crop_size:
            raise ValueError("Image size is too small for cropping.")

        # 随机选择起始点
        start_x = np.random.randint(0, w - crop_size)
        start_y = np.random.randint(0, h - crop_size)

        # 裁剪图像
        image = image[:,:,start_y:start_y+crop_size, start_x:start_x+crop_size]
        print(image.dtype)
        print(image.shape)
        np.save("input.npy", image.permute(0, 2, 3, 1).cpu())
        
        # opal_bnn
        feats = []
        for xi in x:
            feat_i = self.feat_extract(xi)
            feats.append(feat_i)
        # feats_angle = []
        # for j in range(4):
        #     feats_tmp = torch.cat(feats[self.use_v * j:self.use_v * (j + 1)], 1)
        #     feats_tmp = self.block3d[j](self.quant(feats_tmp))
        #     feats_angle.append(self.fuse3d[j](feats_tmp))
        # cv = torch.cat(feats_angle, 1)
        feats_angle = []
        for j in range(4):
            feats_tmp = torch.cat(feats[self.use_v * j:self.use_v * (j + 1)], 1)
            feats_angle.append(self.feat_fuse(feats_tmp))
        cv = torch.cat(feats_angle, 1)
        cv = self.feat_residual(cv)
        prob = self.relu3(cv)
        disp_raw = self.disparitygression(prob) 
        
        
        # resnet_add
        # cv = self.conv1(x[0])
        # cv = self.fuse2d[1](self.quant1(self.fuse2d[0](cv)) + self.quant1(cv))
        # cv = self.fuse2d[3](self.quant2(self.fuse2d[2](cv)) + self.quant2(cv))
        # cv = self.fuse2d[5](self.quant3(self.fuse2d[4](cv)) + self.quant3(cv))
        # cv = self.fuse2d[7](self.quant4(self.fuse2d[6](cv)) + self.quant4(cv))
        # prob = self.fuse3(cv)
        # disp_raw = self.disparitygression(prob)

        
        # concat
        # cv = torch.cat(x, 1)
        # cv = self.conv1(cv)
        # for j in range(4):
        #     #cv = self.fuse2d[j * 2 + 1](self.quant(self.fuse2d[j * 2](cv)) + self.quant(cv))
        #     cv = self.quant(self.fuse2d[j](cv)) + self.quant(cv)
        # prob = self.relu3(self.fuse3(cv))   # 经过fuse3以及relu3操作后得到了概率
        # disp_raw = self.disparitygression(prob) 
        
        
        # residual
        # cv = self.conv1(x[0])
        # for j in range(4):
        #     #cv = self.fuse2d[j * 2 + 1](self.quant(self.fuse2d[j * 2](cv)) + self.quant(cv))
        #     cv = self.quant(self.fuse2d[j](cv)) + self.quant(cv)
        # prob = self.relu3(self.fuse3(cv))   # 经过fuse3以及relu3操作后得到了概率
        # disp_raw = self.disparitygression(prob)     # 将概率进行加权和之后得到初始视差图
        
        
        # feats = []
        # for xi in x:
        #     feat_i = self.feat_extract(xi)
        #     feats.append(feat_i)
        # feats_angle = []
        # for j in range(4):
        #     feats_tmp = torch.stack(feats[self.use_v * j:self.use_v * (j + 1)], dim=2)
        #     feats_tmp = rearrange(feats_tmp, 'b c u h w -> b (c u) h w')
        #     feats_tmp = self.block3d[j](feats_tmp)
        #     #feats_tmp = torch.squeeze(feats_tmp, 2)
        #     feats_angle.append(self.fuse3d[j](feats_tmp))
        # cv = torch.cat(feats_angle, 1)
        # for j in range(4):
        #     cv = self.fuse2d[j * 2 + 1](self.fuse2d[j * 2](cv) + cv)
        # prob = self.relu3(self.fuse3(cv))   # 经过fuse3以及relu3操作后得到了概率
        # disp_raw = self.disparitygression(prob)     # 将概率进行加权和之后得到初始视差图
        # torch.cuda.synchronize()
        # end4 = time.time()
        # 首先定义了四个空列表
        occolusion = []
        disp = []   # 视差图
        mask = []
        raw_warp_img = []   # 初始的变换视图
        for j in range(self.n_angle):   # 这段代码是一个 for 循环，用于处理多个角度的数据。在每次循环中，根据当前角度 j，进行一系列操作，不过此处只取了水平方向和垂直方向的子孔径图像以及相对应的索引
            warpped_views = self.warp(disp_raw, x[self.use_v * j:self.use_v * (j + 1)], IDX[j])
            if self.is_train:
                raw_warp_img.append(warpped_views)
        #     disp_final = self.finetune(disp_raw, x[4])
        #     disp.append(disp_final)
        # disp = torch.cat(disp, 1)
        # disp_mean = torch.mean(disp, 1, True)
        
        return disp_raw, raw_warp_img

    def warp(self, disp, views_list, idx):
        B, C, H, W = views_list[0].shape
        x, y = torch.arange(0, H), torch.arange(0, W)   # 这行代码创建了两个张量 x 和 y，它们分别包含了从0到H-1和从0到W-1的整数序列。
        # 具体来说，torch.arange(0, H)将生成一个从0到H-1的整数序列，而torch.arange(0, W)将生成一个从0到W-1的整数序列。
        # 这种操作通常用于创建一个表示图像坐标的网格，在进行图像处理和计算时经常会用到这样的坐标网格。在这种情况下，x 表示图像的高度方向坐标，而 y 则表示图像的宽度方向坐标。
        self.meshgrid = torch.stack(torch.meshgrid(x, y), -1).unsqueeze(0)  # 1*H*W*2   torch.meshgrid(x, y) 生成了两个张量 x 和 y 的网格坐标，得到的结果是两个形状为 (H, W) 的张量，表示所有可能的坐标对，即每一个元素值都为相对应的坐标，提供的是像素的位置信息。
        # torch.stack 函数将这两个张量在最后一个维度进行堆叠，生成一个新的三维张量。接着，通过 unsqueeze(0) 对结果进行维度扩展，使其在最前面添加一个新的维度，从而得到一个形状为 (1, H, W, 2) 的四维张量。
        disp = disp.squeeze(1)  # 为什么在视察回归中将这一维度扩展出来，现在又要去除？
        # assert H==self.patch_size and W==self.patch_size,"size is different!"
        tmp = []    # 创建了一个空列表 tmp，用于临时保存数据或中间结果。
        meshgrid = self.meshgrid.repeat(B, 1, 1, 1).to(disp)  # B*H*W*2      self.meshgrid.repeat(B, 1, 1, 1) 会将 self.meshgrid 沿着各个维度重复指定的次数，.to(disp) 表示将重复后的张量转换为和 disp 张量相同的数据类型和设备类型。
        for k in range(9):  # 此处的0~8应该指的就是某个方向的9张子孔径图像
            u, v = divmod(idx[k], 9)    # 将 idx[k] 除以 9 求商和余数，得到的商赋值给 u，余数赋值给 v。此处的u，v就是子孔径图像的uv维度，u代表行，v代表列
            grid = torch.stack([
                torch.clip(meshgrid[:, :, :, 1] - disp * (v - 4), 0, W - 1),    # torch.clip 函数用于对张量进行截断操作，将张量的值限制在指定的范围内。
                torch.clip(meshgrid[:, :, :, 0] - disp * (u - 4), 0, H - 1)     # 此处有点像论文中的公式（2）
            ], -1) / (W - 1) * 2 - 1  # B*H*W*2  归一化到-1，1
            tmp.append(grid_sample(views_list[k], grid, align_corners=True))
        return tmp
    # 这段代码是一个 warp 方法，根据输入的视差（disp）、视角列表（views_list）和索引（idx），将视角列表中的图像进行视角校正（warp）操作。
    # 首先，代码获取视角列表中第一个图像的大小，并创建了一个网格矩阵 meshgrid，其形状为 [1, H, W, 2]，其中 H 和 W 是图像的高度和宽度。
    # 然后，对输入的视差 disp 进行了挤压操作，将其在维度1上去除。接下来，通过重复 meshgrid 来扩展为与视角列表 views_list 相同的形状，得到 meshgrid 的形状变为 [B, H, W, 2]，其中 B 是 views_list 的批量大小。
    # 接下来，通过循环遍历索引 idx，将其拆分为 u 和 v，并根据视差调整 meshgrid，从而校正每个视角的图像。在这里，使用了 torch.clip 函数将调整后的坐标限制在图像范围内，并进行归一化和映射到[-1,1]的操作，得到了最终的校正后的采样点 grid。
    # 最后，使用 grid_sample 函数对每个视角的图像进行采样，根据调整后的 grid 从原始图像中提取对应的像素值，存储在临时列表 tmp 中，并返回这个列表作为结果。

    # 这段代码实现了对输入张量进行视差回归的操作，即根据输入的视差值范围，计算出每个像素点在不同视差值下的加权和，并返回这些加权和组成的张量。
    def disparitygression(self, input):
        disparity_values = torch.linspace(-4, 4, 9, device=self.device)
        x = disparity_values.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)   # 将原始的一维张量转换为了一个四维张量    unsqueeze(0) 将在第0维度（最外层维度）上插入一个新维度，将原本的一维张量变成了二维张量。
        out = torch.sum(torch.multiply(input, x), 1)
        return out.unsqueeze(1)     # 这行代码对变量 out 进行了 unsqueeze(1) 操作，将其在第1维度（第二个维度）上进行扩展。这样操作会在 out 的维度中插入一个新的维度，使得输出的张量维度数量增加了一维。
    # 这段代码实现了一个名为 disparityRegression 的方法。该方法接受一个输入张量 input（概率），并对其进行视差回归（disparity regression）的操作。
    # 首先，代码使用 torch.linspace 生成了一个包含从-4到4的等间距数字的张量 disparity_values，总共包含了9个值。这些值代表了视差的取值范围。
    # 接下来，代码对 disparity_values 进行了一系列的 unsqueeze 操作，将其形状调整为 (1, 9, 1, 1)，以便与输入张量 input 进行相乘操作。
    # 然后，代码使用 torch.multiply 函数对 input 和 x 进行逐元素相乘操作，并通过 torch.sum 对逐元素相乘的结果在第1维上进行求和操作，得到了一个张量 out。
    # 最后，代码将 out 的形状通过 unsqueeze(1) 进行调整，得到最终的输出张量，并返回这个张量作为方法的输出结果。
    def cal_occlusion(self, views):
        views = torch.stack(views, -1)  # B*C*H*W*9
        if self.grad_v == 9:
            grad = torch.mean(torch.abs(views[:, :, :, :, 1:] - views[:, :, :, :, :8]), dim=[1, 4])
        elif self.grad_v == 5:
            grad = torch.mean(torch.abs(views[:, :, :, :, 3:7] - views[:, :, :, :, 2:6]), dim=[1, 4])  # B*H*W
        return grad.unsqueeze(1)


# class OPENetmodel(BaseModel):  # Basemodel
#     @staticmethod
#     def modify_commandline_options(parser, is_train=True):
#         return parser
#
#     def __init__(self, opt):
#         BaseModel.__init__(self, opt)
#         self.loss_names = ['L1']
#         if opt.losses.find(
#                 'smooth') != -1:  # 判断 opt.losses 字符串中是否包含子字符串 'smooth'，如果包含则向 self.loss_names 列表中添加一个元素 'smoothness'。
#             self.loss_names.append('smoothness')
#         self.visual_names = ['center_input', 'output', 'label']
#
#         self.model_names = ['EPI']
#         net = eval(self.opt.net_version)(opt, self.device,
#                                          self.isTrain)  # eval(self.opt.net_version)的作用是返回网络版本，即下面的Unsup31_15_53、Unsup27_16_16或Unsup26_21_56
#         # net = Unsup31_15_53(opt, self.device,self.isTrain)   # final syth:07-13-39  09-10-27(增强) real:08-1-12
#         # net = Unsup27_16_16(opt, self.device,self.isTrain) # finetune syth:07-13-42
#         # net = Unsup26_21_56(opt, self.device,self.isTrain) # fast syth:07-13-44  07-20-46  real: 08-01-23
#
#         self.netEPI = init_net(net, opt.init_type, opt.init_gain, self.gpu_ids)
#         self.use_v = opt.use_views  # 表示9*9的子孔径图像，索引为0~8（一行或者一列），use_views为9
#
#         self.center_index = self.use_v // 2  # 将use_views的一半作为中心视图的索引：4
#         self.alpha = opt.alpha
#         self.pad = opt.pad
#         self.lamda = opt.lamda
#
#         self.test_loss_log = 0  # 将初始测试损失值设为 0。
#         self.test_loss = torch.nn.L1Loss()  # 创建一个 L1 损失函数对象
#         if self.isTrain:
#             # define loss functions
#             self.criterionL1 = eval(self.opt.loss_version)(self.opt, self.device)
#             self.optimizer = get_optimizer(opt, filter(lambda p: p.requires_grad, self.netEPI.parameters()), LR=opt.lr)
#             self.optimizers.append(self.optimizer)
#         # 这段代码片段展示了在训练模式下（self.isTrain 为真时）进行的操作：
#         # self.criterionL1 = eval(self.opt.loss_version)(self.opt, self.device): 根据 self.opt.loss_version 的值，动态地创建了一个损失函数对象，并将其赋值给
#         # self.criterionL1。通过 eval() 函数执行字符串表示的表达式，这里假设 self.opt.loss_version 是一个字符串，代表了要使用的损失函数的名称或类型。
#         # self.optimizer = get_optimizer(opt, filter(lambda p: p.requires_grad, self.netEPI.parameters()), LR=opt.lr): 调用 get_optimizer 函数来获取一个优化器对象，
#         # 该函数接受一些参数，包括模型的可训练参数（通过 filter(lambda p: p.requires_grad, self.netEPI.parameters()) 筛选出需要梯度更新的参数）、学习率等参数。最后，将获取的优化器对象赋值给 self.optimizer。
#         # lambda p: p.requires_grad 是一个匿名函数，用于判断参数 p 是否需要计算梯度（即 requires_grad 属性为 True）。self.netEPI.parameters() 返回模型 self.netEPI 的所有参数。
#         # 通过 filter 函数，仅保留那些需要计算梯度的参数，最终返回一个迭代器或列表，其中包含需要梯度更新的参数对象。
#         # self.optimizers.append(self.optimizer): 将创建的优化器对象 self.optimizer 添加到 self.optimizers 列表中，这可能是为了在训练过程中管理多个优化器。
#         # 这些操作通常在模型训练阶段用于设置损失函数和优化器，以便在训练循环中使用它们来计算损失并更新模型参数。
#
#     def set_input(self, inputs, epoch):
#         self.epoch = epoch
#         # self.supervise_view = rearrange(inputs[0].to(self.device), 'b c (h1 h) (w1 w) u v -> (b h1 w1) c h w u v', h1=8, w1=8)
#         self.supervise_view = inputs[0].to(
#             self.device)  # 这段代码将输入数据 inputs[0] 移动（或复制）到指定的设备上，即将数据传输到 GPU 或 CPU 上进行处理。具体来说，inputs[0] 是模型的输入数据，通过调用 .to(self.device) 方法，将其移动到在类中定义的 self.device 上。
#         # 移动数据到特定设备上的操作通常用于确保模型在训练或推理过程中能够利用 GPU 等加速器来加快计算速度。通过在不同设备上存储和处理数据，可以充分利用硬件资源，提高模型的性能表现。
#         self.input = []
#         for j in range(self.use_v):  # 对 self.supervise_view 数据进行切片操作，选择特定的数据子集。具体来说，self.supervise_view[:,:,:,:, self.center_index,j] 使用了多维切片操作，其中 : 表示选择该维度上的所有元素，self.center_index 和 j 则表示在最后两个维度上选择特定的索引。
#             self.input.append(self.supervise_view[:, :, :, :, self.center_index, j])  # 行
#         for j in range(self.use_v):
#             self.input.append(self.supervise_view[:, :, :, :, j, self.center_index])  # 列
#         for j in range(self.use_v):
#             self.input.append(self.supervise_view[:, :, :, :, j, j])  # 对角线
#         for j in range(self.use_v):
#             self.input.append(self.supervise_view[:, :, :, :, j, self.use_v - 1 - j])  # 反对角线（此处应该为self.use_v-j吧）
#         # ------------输入是来自四个不同方向的子孔径图像---------------
#         self.center_input = self.input[self.center_index]
#         self.label = inputs[1].to(self.device)
#         # 这段代码是一个方法 set_input，用于设置模型的输入数据。下面是这个方法的主要步骤：
#         # 将输入中的第一个元素 inputs[0] 移动到设备（如 GPU）上，然后赋值给 self.supervise_view。注释部分的代码使用了 rearrange 函数对输入进行了一些重排操作，但实际上并没有被使用。
#         # 初始化空列表 self.input 用来存储处理后的输入数据。
#         # 通过循环，从 self.supervise_view 中提取特定位置的数据，并将其添加到 self.input 列表中。具体来说，根据 center_index 和 use_v 的值，从 self.supervise_view 中选择不同位置的数据添加到 self.input 中。
#         # 然后，将 self.input 中的第 center_index 个元素赋值给 self.center_input。
#         # 最后，将输入中的第二个元素 inputs[1] 移动到设备上，并赋值给 self.label。
#         # 这个方法的作用是根据输入数据中的特定位置信息，构建模型的输入数据，为模型提供正确的数据格式以及需要关注的信息，同时准备好模型的标签数据。这样有助于模型在训练和推理过程中正确地处理输入数据。
#
#     # self.netEPI(self.input) 表示将输入数据 self.input 传递给网络模型 self.netEPI 进行计算，得到的结果赋值给 self.output 和 self.raw_warp_img。
#     # self.output 是模型的输出结果，self.raw_warp_img 是网络模型中间的原始扭曲图像。
#     def forward(self):  #############################################################################
#         """Run forward pass; called by both functions <optimize_parameters> and <test>."""
#         self.output, self.raw_warp_img = self.netEPI(self.input)  # G(A)
#         self.test_loss_log = 0
#
#     def backward_G(self):
#         # if self.epoch <= self.opt.n_epochs:
#         #     self.loss_L1 = self.criterionL1(self.output, self.input[:9], self.input[9:18])
#         # else:
#         #     self.loss_L1 = self.criterionL1_5(self.output, self.input[:9], self.input[9:18])
#         # self.loss_L1 = self.criterionL1(self.output, self.input[:9], self.input[9:18])
#         self.loss_L1 = self.criterionL1(self.epoch, self.output, self.input[:9],
#                                         self.input[9:18])  # OPENet-fast，self.input[:9], self.input[9:18]分别代表水平喝垂直方向
#         # 通常 L1 损失是指 Mean Absolute Error（MAE），用于衡量模型输出与目标数据之间的绝对差异。
#         # self.epoch 可能是当前训练的 epoch 数，用于在损失函数中引入一些动态调整或变化；self.output 是模型的输出结果；
#         # 该行代码的作用是调用损失函数 self.criterionL1，并将相关参数传递给该函数，计算得到 L1 损失值。
#         if self.raw_warp_img is None:
#             self.loss_total = self.loss_L1  # 此处的raw_warp_img指什么
#         else:
#             self.loss_raw = 0
#             for views in self.raw_warp_img:
#                 self.loss_raw += self.criterionL1(self.epoch, views)
#             self.loss_total = 0.6 * self.loss_L1 + 0.4 * self.loss_raw  # 从此处可以看出loss_L1是论文中raw损失，loss_raw是final损失
#             # self.loss_total = self.loss_L1
#         self.loss_smoothness = get_smooth_loss(self.output, self.center_input, self.lamda)
#
#         if 'smoothness' in self.loss_names and self.epoch > 2 * self.opt.n_epochs:
#             self.loss_total += self.loss_smoothness
#         self.loss_total.backward()
#         # 这段代码实现了模型的反向传播 backward_G()。该方法用于计算模型参数的梯度，并更新参数。
#         # 具体来说，第一步是计算损失函数 self.loss_total，其中包括两部分：L1 损失 self.loss_L1 和原始扭曲图像损失 self.loss_raw，二者的权重比例为 0.6 和 0.4。
#         # L1 损失 self.loss_L1 衡量了模型输出结果与真实数据之间的差异；原始扭曲图像损失 self.loss_raw 衡量了网络模型中间产生的误差。同时还计算了平滑性损失 self.loss_smoothness，该损失用于缓解输出结果的过度平滑现象。
#         # 接下来，将 self.loss_total 反向传播，并计算参数的梯度。反向传播可以自动计算参数的梯度，并存储在模型中。在优化过程中，使用梯度下降等方法基于梯度更新模型的参数，以优化损失函数值并提升模型的性能表现。
#
#     def optimize_parameters(self):
#         self.netEPI.train()
#         self.forward()
#         self.optimizer.zero_grad()  # 将优化器中所有参数的梯度清零，以准备接收新一轮的梯度计算。
#         self.backward_G()
#         self.optimizer.step()  # 根据优化器中存储的参数梯度信息，更新模型的参数。
#
#
#
#
#
# # finetune 1
# # views 36  指来自四个不同方向的所有子孔径图像，共36张图像
# # cycle 1
# # transformer 0
# class Unsup27_16_16(nn.Module):  # v3
#     def __init__(self, opt, device, is_train=True):
#         super().__init__()
#         self.is_train = is_train  # 设置是否为训练模式，默认为 True。
#         self.n_angle = 2  # 角度数量为 2，可能与后续涉及视角相关的操作有关。
#         feats = 64  # 特征通道数为 64，后续会根据需要调整。
#         self.device = device
#         self.use_v = opt.use_views  # 使用的视角数，根据外部传入的参数确定。
#         self.grad_v = opt.grad_v  # 梯度计算使用的视角数，同样根据外部参数确定。
#         self.feat_extract = Feature(in_channels=opt.input_c, out_channels=8)    # 这段代码创建了一个名为 feat_extract 的特征提取器（Feature），并指定了输入通道数和输出通道数。
#         self.block3d = nn.ModuleList()  # 这行代码创建了一个空的 nn.ModuleList() 类型的列表，命名为 self.block3d。在 PyTorch 中，nn.ModuleList 是一个存储 nn.Module 的列表的容器，通常用于存储神经网络模型中的子模块。
#         self.fuse3d = nn.ModuleList()
#         for _ in range(4):  # 3D 卷积块和融合块的列表，共有 4 个。这段代码是一个神经网络模型的构建过程，使用了 PyTorch 中的 nn.Sequential 模块来按顺序组合多个神经网络层。
#             self.block3d.append(    # 每个 3D 卷积块由一个 nn.Sequential 模块组成，其中包含了一个 3D 卷积层 (nn.Conv3d) 和一个 LeakyReLU 激活函数 (nn.LeakyReLU)。这些层被配置为接收输入大小为 (8, H, W) 的张量并输出大小为 (64, H, W) 的张量。
#                 nn.Sequential(nn.Conv3d(8, 64, (self.use_v, 3, 3), 1, padding=(0, 1, 1)), nn.LeakyReLU(0.2, True)))  # 卷积核大小为 (self.use_v, 3, 3)，填充操作为在第二和第三维度上分别填充 1 个单位，保持输入输出大小相同。
#             self.fuse3d.append(nn.Sequential(nn.Conv2d(64, 32, 1, 1), nn.LeakyReLU(0.2, True)))  # Leaky ReLU 激活层，负斜率参数为 0.2，inplace 参数为 True，表示进行原地操作
#         # 每个融合块由一个 nn.Sequential 模块组成，其中包含了一个 2D 卷积层 (nn.Conv2d) 和一个 LeakyReLU 激活函数 (nn.LeakyReLU)。这些层被配置为接收输入大小为 (64, H, W) 的张量并输出大小为 (32, H, W) 的张量。
#         feats *= 2
#         self.fuse2d = nn.ModuleList()
#         for j in range(4):  # 2D 融合块的列表，包含了一系列卷积和 LeakyReLU 操作。
#             self.fuse2d.append(nn.Sequential(nn.Conv2d(feats, feats, 3, 1, padding=1), nn.LeakyReLU(0.2, True),     # 卷积核大小为 3x3，步长为 1
#                                              nn.Conv2d(feats, feats, 1, 1)))    # 卷积核大小为 1x1，步长为 1
#             self.fuse2d.append(nn.LeakyReLU(0.2, True))     # 向 self.fuse2d 中添加了一个单独的 LeakyReLU 激活层(nn.LeakyReLU)
#         self.fuse3 = nn.Conv2d(feats, 9, 3, 1, padding=1)  # 最终的卷积层，输出通道数为 9(对应视差范围中的9个视差值)。   这段代码定义了一个 2D 卷积层，命名为 fuse3
#         self.relu3 = nn.Softmax(dim=1)  # Softmax 激活函数，用于产生概率分布。    这段代码定义了一个 Softmax 激活层，命名为 relu3
#         # dim=1：表示在维度为 1 的轴上计算 Softmax 函数。这意味着对输入数据的第一个维度（通常是特征维度）进行 Softmax 操作，将每个类别的分数转换为概率值。
#         # Softmax 操作通常用于多分类问题中，将神经网络输出的原始分数（logits）转换为概率分布，使得每个类别的输出值在 [0, 1] 范围内，并且所有类别的概率之和为 1。
#         self.finetune = ContextAdjustmentLayer()  # 上下文调整层，可能用于模型微调或优化。
#
#     # 对输入 x 中的每个样本 xi 进行特征提取，得到特征 feats。
#     # 将提取的特征 feats 进行处理，得到一组 feats_angle。
#     # 将 feats_angle 组合成一个张量 cv，然后经过一系列操作，最终得到输出概率 prob 和视差 disp_raw。
#     # 根据 disp_raw 进行视差值优化和遮挡计算，得到最终的视差结果 disp_mean，并将中间结果存储在 raw_warp_img 中返回。
#     def forward(self, x):   # 输入是来自四个不同方向的子孔径图像
#         feats = []
#         for xi in x:
#             feat_i = self.feat_extract(xi)  # 首先经过特征提取层处理
#             feats.append(feat_i)
#         # torch.cuda.synchronize()
#         # end1 = time.time()
#         feats_angle = []    # 接下来进行 EPI-Trans 处理，即代码中的block3d和fuse3d处理
#         for j in range(4):  # 四个不同方向的特征分别进行一次处理然后添加到 feats_angle        下面一行的9应该指的就是在dim=2维度上，将每一个方向上的9张子孔径图像进行堆叠，形成一个具有更多维度的张量，即将9个B*8*h*w的张量堆叠为1个B*8*9*h*w的张量
#             feats_tmp = torch.stack(feats[self.use_v * j:self.use_v * (j + 1)], dim=2)  # B*8*9*h*w     选择列表 feats 中从索引 self.use_v * j 到 self.use_v * (j + 1) - 1 的特征张量，然后在 dim=2 上进行堆叠。
#             feats_tmp = self.block3d[j](feats_tmp)  # 将 feats_tmp 输入到名为 block3d 的第 j 个块中进行处理，从8通道变为64通道。
#             feats_tmp = torch.squeeze(feats_tmp, 2)  # B*64*h*w 在feats_tmp张量的第三个维度上进行挤压操作，即将维数为1的维度去除，使得张量更加紧凑和易于处理。在深度学习模型中，对张量进行挤压操作可以帮助减少不必要的维度，以符合后续操作的要求，同时节省内存空间和计算资源。
#             feats_angle.append(self.fuse3d[j](feats_tmp))   # 将挤压后的特征张量 feats_tmp 输入到名为 fuse3d 的第 j 个融合块中，并将32通道的结果添加到 feats_angle 中。
#         cv = torch.cat(feats_angle, 1)      # 沿着维度 1（dim=1）将张量序列 feats_angle 进行连接，生成一个新的128通道的张量。此处的拼接操作不就是将EPI-Trans的4个不同方向的结果进行拼接么
#         for j in range(4):  # 这段代码的作用是进行反复的特征融合操作。这种多次融合的操作可以帮助特征更好地交互和融合，从而增强特征的表达能力，使得网络能够学习到更加丰富和复杂的特征表示。这对于图像处理和计算机视觉任务中的特征提取和表示学习非常有帮助。
#             cv = self.fuse2d[j * 2 + 1](self.fuse2d[j * 2](cv) + cv)
#         # torch.cuda.synchronize()
#         # end3 = time.time()
#         prob = self.relu3(self.fuse3(cv))   # 经过fuse3以及relu3操作后得到了概率
#         disp_raw = self.disparitygression(prob)     # 将概率进行加权和之后得到初始视差图
#         # torch.cuda.synchronize()
#         # end4 = time.time()
#         # 首先定义了四个空列表
#         occolusion = []
#         disp = []   # 视差图
#         mask = []
#         raw_warp_img = []   # 初始的变换视图
#         for j in range(self.n_angle):   # 这段代码是一个 for 循环，用于处理多个角度的数据。在每次循环中，根据当前角度 j，进行一系列操作，不过此处只取了水平方向和垂直方向的子孔径图像以及相对应的索引
#             warpped_views = self.warp(disp_raw, x[self.use_v * j:self.use_v * (j + 1)], IDX[j])
#             if self.is_train:
#                 raw_warp_img.append(warpped_views)
#             occu = self.cal_occlusion(warpped_views)
#
#             disp_final, occu_final = self.finetune(disp_raw, occu, x[4])    # x[4]表示中心视图
#             # mask.append(torch.where(disp_final<0.03,1,0).float())
#             disp.append(disp_final)
#             # occolusion.append(occu_final)
#         # mask = torch.where(torch.mean(torch.cat(mask, 1),1, True) < 1 ,0., 1.,) # all view==1, mask=1
#         disp = torch.cat(disp, 1)
#         disp_mean = torch.mean(disp, 1, True)   # 此处的平均视差图就是将多个视角的视差求平均
#         # 这两行代码的作用是对列表中的多个张量进行拼接，然后计算拼接后张量沿指定维度的平均值。
#         return disp_mean, raw_warp_img
#
#     def warp(self, disp, views_list, idx):
#         B, C, H, W = views_list[0].shape
#         x, y = torch.arange(0, H), torch.arange(0, W)   # 这行代码创建了两个张量 x 和 y，它们分别包含了从0到H-1和从0到W-1的整数序列。
#         # 具体来说，torch.arange(0, H)将生成一个从0到H-1的整数序列，而torch.arange(0, W)将生成一个从0到W-1的整数序列。
#         # 这种操作通常用于创建一个表示图像坐标的网格，在进行图像处理和计算时经常会用到这样的坐标网格。在这种情况下，x 表示图像的高度方向坐标，而 y 则表示图像的宽度方向坐标。
#         self.meshgrid = torch.stack(torch.meshgrid(x, y), -1).unsqueeze(0)  # 1*H*W*2   torch.meshgrid(x, y) 生成了两个张量 x 和 y 的网格坐标，得到的结果是两个形状为 (H, W) 的张量，表示所有可能的坐标对，即每一个元素值都为相对应的坐标，提供的是像素的位置信息。
#         # torch.stack 函数将这两个张量在最后一个维度进行堆叠，生成一个新的三维张量。接着，通过 unsqueeze(0) 对结果进行维度扩展，使其在最前面添加一个新的维度，从而得到一个形状为 (1, H, W, 2) 的四维张量。
#         disp = disp.squeeze(1)  # 为什么在视察回归中将这一维度扩展出来，现在又要去除？
#         # assert H==self.patch_size and W==self.patch_size,"size is different!"
#         tmp = []    # 创建了一个空列表 tmp，用于临时保存数据或中间结果。
#         meshgrid = self.meshgrid.repeat(B, 1, 1, 1).to(disp)  # B*H*W*2      self.meshgrid.repeat(B, 1, 1, 1) 会将 self.meshgrid 沿着各个维度重复指定的次数，.to(disp) 表示将重复后的张量转换为和 disp 张量相同的数据类型和设备类型。
#         for k in range(9):  # 此处的0~8应该指的就是某个方向的9张子孔径图像
#             u, v = divmod(idx[k], 9)    # 将 idx[k] 除以 9 求商和余数，得到的商赋值给 u，余数赋值给 v。此处的u，v就是子孔径图像的uv维度，u代表行，v代表列
#             grid = torch.stack([
#                 torch.clip(meshgrid[:, :, :, 1] - disp * (v - 4), 0, W - 1),    # torch.clip 函数用于对张量进行截断操作，将张量的值限制在指定的范围内。
#                 torch.clip(meshgrid[:, :, :, 0] - disp * (u - 4), 0, H - 1)     # 此处有点像论文中的公式（2）
#             ], -1) / (W - 1) * 2 - 1  # B*H*W*2  归一化到-1，1
#             tmp.append(grid_sample(views_list[k], grid, align_corners=True))
#         return tmp
#     # 这段代码是一个 warp 方法，根据输入的视差（disp）、视角列表（views_list）和索引（idx），将视角列表中的图像进行视角校正（warp）操作。
#     # 首先，代码获取视角列表中第一个图像的大小，并创建了一个网格矩阵 meshgrid，其形状为 [1, H, W, 2]，其中 H 和 W 是图像的高度和宽度。
#     # 然后，对输入的视差 disp 进行了挤压操作，将其在维度1上去除。接下来，通过重复 meshgrid 来扩展为与视角列表 views_list 相同的形状，得到 meshgrid 的形状变为 [B, H, W, 2]，其中 B 是 views_list 的批量大小。
#     # 接下来，通过循环遍历索引 idx，将其拆分为 u 和 v，并根据视差调整 meshgrid，从而校正每个视角的图像。在这里，使用了 torch.clip 函数将调整后的坐标限制在图像范围内，并进行归一化和映射到[-1,1]的操作，得到了最终的校正后的采样点 grid。
#     # 最后，使用 grid_sample 函数对每个视角的图像进行采样，根据调整后的 grid 从原始图像中提取对应的像素值，存储在临时列表 tmp 中，并返回这个列表作为结果。
#
#     # 这段代码实现了对输入张量进行视差回归的操作，即根据输入的视差值范围，计算出每个像素点在不同视差值下的加权和，并返回这些加权和组成的张量。
#     def disparitygression(self, input):
#         disparity_values = torch.linspace(-4, 4, 9, device=self.device)
#         x = disparity_values.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)   # 将原始的一维张量转换为了一个四维张量    unsqueeze(0) 将在第0维度（最外层维度）上插入一个新维度，将原本的一维张量变成了二维张量。
#         out = torch.sum(torch.multiply(input, x), 1)
#         return out.unsqueeze(1)     # 这行代码对变量 out 进行了 unsqueeze(1) 操作，将其在第1维度（第二个维度）上进行扩展。这样操作会在 out 的维度中插入一个新的维度，使得输出的张量维度数量增加了一维。
#     # 这段代码实现了一个名为 disparityRegression 的方法。该方法接受一个输入张量 input（概率），并对其进行视差回归（disparity regression）的操作。
#     # 首先，代码使用 torch.linspace 生成了一个包含从-4到4的等间距数字的张量 disparity_values，总共包含了9个值。这些值代表了视差的取值范围。
#     # 接下来，代码对 disparity_values 进行了一系列的 unsqueeze 操作，将其形状调整为 (1, 9, 1, 1)，以便与输入张量 input 进行相乘操作。
#     # 然后，代码使用 torch.multiply 函数对 input 和 x 进行逐元素相乘操作，并通过 torch.sum 对逐元素相乘的结果在第1维上进行求和操作，得到了一个张量 out。
#     # 最后，代码将 out 的形状通过 unsqueeze(1) 进行调整，得到最终的输出张量，并返回这个张量作为方法的输出结果。
#
#     # 这行代码的作用是计算相邻视角之间的梯度信息，并将平均值存储在 grad 张量中，用于后续的处理或计算。
#     # views[:, :, :, :, 1:] - views[:, :, :, :, :8]：这部分代码首先对 views 张量进行切片操作，
#     # 从第一个视角到第八个视角（共8个），然后再从第二个视角到第九个视角（共8个），这样操作得到的结果是相邻视角之间的差值张量，形状为 B*C*H*W*8。
#     # torch.abs()：对上一步得到的差值张量取绝对值，将所有元素转换为非负值。
#     # torch.mean(..., dim=[1, 4])：接着，通过 torch.mean() 函数计算绝对差值张量在第1和第4维度上的平均值，即对通道维度和视角维度求平均值。
#     # 最终得到的 grad 张量形状为 B*H*W，其中每个像素值代表了相邻视角之间的平均梯度值。
#     def cal_occlusion(self, views):     # 遮挡部分的计算怎么完全对应不上论文？
#         views = torch.stack(views, -1)  # B*C*H*W*9
#         if self.grad_v == 9:    # views[:, :, :, :, 1:] 表示去掉了第一个视角的像素值，views[:, :, :, :, :8] 表示只取0~7视角的像素值
#             grad = torch.mean(torch.abs(views[:, :, :, :, 1:] - views[:, :, :, :, :8]), dim=[1, 4])  # 指定了在第 1 和第 4 维度上进行求均值
#         elif self.grad_v == 5:
#             grad = torch.mean(torch.abs(views[:, :, :, :, 3:7] - views[:, :, :, :, 2:6]), dim=[1, 4])  # B*H*W
#         return grad.unsqueeze(1)
#     # 这段代码是一个名为 cal_occlusion 的方法，用于计算视差图中的遮挡情况。这个方法接受一个名为 views 的列表作为输入，表示不同视角下的视差图。在方法内部，
#     # 首先使用 torch.stack(views, -1) 对输入的视差图进行堆叠，得到一个新的张量，其形状为 B*C*H*W*9，其中 B 代表 batch size，C 代表通道数，H 和 W 分别代表高度和宽度，9 表示有 9 个视角。
#     # 接着，代码根据 self.grad_v 的取值（可能是 9 或者 5）来计算视差图中的梯度信息。如果 self.grad_v 等于 9，那么代码计算每个像素对应的视差值的平均绝对梯度；
#     # 如果 self.grad_v 等于 5，那么代码计算每个像素对应的视差值的局部平均绝对梯度。最终得到的 grad 张量的形状为 B*H*W。
#     # 最后，代码通过 unsqueeze(1) 操作在 grad 张量的第1维上进行扩展，得到一个形状为 B*1*H*W 的输出张量，并将这个张量作为方法的返回结果。


#下面这一部分代码的不同之处在于，在 forward 中计算出初始视差图就结束了，并未进行后续的warp以及cal_occlusion操作
class Unsup27_16_16_nof(nn.Module):  # v3
    def __init__(self, opt, device, is_train=True):
        super().__init__()
        self.is_train = is_train
        self.n_angle = 2
        feats = 64
        self.device = device
        self.use_v = opt.use_views
        self.grad_v = opt.grad_v
        self.feat_extract = Feature(in_channels=opt.input_c, out_channels=8)

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
        self.finetune = ContextAdjustmentLayer()

    def forward(self, x):
        feats = []

        for xi in x:
            feat_i = self.feat_extract(xi)
            feats.append(feat_i)

        feats_angle = []
        for j in range(4):
            feats_tmp = torch.stack(feats[self.use_v * j:self.use_v * (j + 1)], dim=2)  # B*8*9*h*w
            feats_tmp = self.block3d[j](feats_tmp)
            feats_tmp = torch.squeeze(feats_tmp, 2)  # B*64*h*w
            feats_angle.append(self.fuse3d[j](feats_tmp))
        cv = torch.cat(feats_angle, 1)
        for j in range(4):
            cv = self.fuse2d[j * 2 + 1](self.fuse2d[j * 2](cv) + cv)

        prob = self.relu3(self.fuse3(cv))
        disp_raw = self.disparitygression(prob)

        return disp_raw, None

    def warp(self, disp, views_list, idx):
        B, C, H, W = views_list[0].shape
        x, y = torch.arange(0, H), torch.arange(0, W)
        self.meshgrid = torch.stack(torch.meshgrid(x, y), -1).unsqueeze(0)  # 1*H*W*2
        disp = disp.squeeze(1)
        tmp = []
        meshgrid = self.meshgrid.repeat(B, 1, 1, 1).to(disp)  # B*H*W*2
        for k in range(9):
            u, v = divmod(idx[k], 9)
            grid = torch.stack([
                torch.clip(meshgrid[:, :, :, 1] - disp * (v - 4), 0, W - 1),
                torch.clip(meshgrid[:, :, :, 0] - disp * (u - 4), 0, H - 1)
            ], -1) / (W - 1) * 2 - 1  # B*H*W*2  归一化到-1，1
            tmp.append(grid_sample(views_list[k], grid, align_corners=True))
        return tmp

    def disparitygression(self, input):
        disparity_values = torch.linspace(-4, 4, 9, device=self.device)
        x = disparity_values.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        out = torch.sum(torch.multiply(input, x), 1)
        return out.unsqueeze(1)

    def cal_occlusion(self, views):
        views = torch.stack(views, -1)  # B*C*H*W*9
        if self.grad_v == 9:
            grad = torch.mean(torch.abs(views[:, :, :, :, 1:] - views[:, :, :, :, :8]), dim=[1, 4])
        elif self.grad_v == 5:
            grad = torch.mean(torch.abs(views[:, :, :, :, 3:7] - views[:, :, :, :, 2:6]), dim=[1, 4])  # B*H*W
        return grad.unsqueeze(1)


# finetune 1
# views 36
# cycle 1
# transformer 1     相比上面的代码多了transformer部分
# final     似乎这一部分才是最完整的代码
class Unsup31_15_53(nn.Module):
    def __init__(self, opt, device, is_train=True):
        super().__init__()
        self.is_train = is_train
        self.n_angle = 2
        feats = 64
        self.device = device
        self.use_v = opt.use_views
        self.grad_v = opt.grad_v
        self.patch_size = opt.input_size
        self.center_index = self.use_v // 2
        # feature
        self.feat_extract = Feature(in_channels=opt.input_c, out_channels=8)
        # transformer
        ## v3 空间到特征
        self.transformer = nn.ModuleList()
        self.block3d = nn.ModuleList()
        self.fuse3d = nn.ModuleList()
        for j in range(4):
            self.transformer.append(transformer_layer(32, 9, 2, 4))  # v2:36 v3:9
            self.block3d.append(
                nn.Sequential(nn.Conv3d(8, 64, (self.use_v, 3, 3), 1, padding=(0, 1, 1)), nn.LeakyReLU(0.2, True)))
            self.fuse3d.append(nn.Sequential(nn.Conv2d(64, 32, 1, 1), nn.LeakyReLU(0.2, True)))
            # regression
        feats = 128
        self.fuse2d = nn.ModuleList()
        for j in range(4):
            self.fuse2d.append(nn.Sequential(nn.Conv2d(feats, feats, 3, 1, padding=1), nn.LeakyReLU(0.2, True),
                                             nn.Conv2d(feats, feats, 1, 1)))
            self.fuse2d.append(nn.LeakyReLU(0.2, True))
        self.fuse3 = nn.Conv2d(feats, 9, 3, 1, padding=1)
        self.relu3 = nn.Softmax(dim=1)
        # finetune
        self.finetune = ContextAdjustmentLayer()

    def forward(self, x):
        feats = []
        # feature
        for xi in x:
            feat_i = self.feat_extract(xi)
            # atten_i = adaptive_avg_pool2d(feat_i, (1,1))
            # feats.append(feat_i*atten_i)
            feats.append(feat_i)

        # transformer
        ## v3 10-31-15-53 分别trans24 分别fuse feats：8-64-32 空间到feats
        feats_angle = []
        for j in range(4):
            feats_tmp = torch.stack(feats[self.use_v * j:self.use_v * (j + 1)], 2)
            feats_tmp = rearrange(feats_tmp, 'b c a (h1 h) (w1 w) -> b (c h1 w1) a h w', h1=2,
                                  w1=2)  # B*32*9*(h/2)*(w/2)
            feats_tmp = self.transformer[j](feats_tmp)
            feats_tmp = rearrange(feats_tmp, 'b (c h1 w1) a h w -> b c a (h1 h) (w1 w)', h1=2,
                                  w1=2)  # B*8*36*(h/2)*(w/2)
            feats_tmp = self.block3d[j](feats_tmp)  # B*32*36*(h/2)*(w/2)
            feats_tmp = torch.squeeze(feats_tmp, 2)
            feats_angle.append(self.fuse3d[j](feats_tmp))

        # regression
        cv = torch.cat(feats_angle, 1)  # b,128,h,w
        for j in range(4):
            cv = self.fuse2d[j * 2 + 1](self.fuse2d[j * 2](cv) + cv)
        prob = self.relu3(self.fuse3(cv))
        disp_raw = self.disparitygression(prob)
        # finetune
        occolusion = []
        disp = []
        mask = []
        raw_warp_img = []
        for j in range(self.n_angle):
            warpped_views = self.warp(disp_raw, x[self.use_v * j:self.use_v * (j + 1)], IDX[j])
            if self.is_train:
                raw_warp_img.append(warpped_views)
            occu = self.cal_occlusion(warpped_views)
            # self.vis(occu, j)

            disp_final, occu_final = self.finetune(disp_raw, occu, x[4])
            mask.append(torch.where(disp_final < 0.03, 1, 0).float())
            disp.append(disp_final)
            # occolusion.append(occu_final)
        mask = torch.where(torch.mean(torch.cat(mask, 1), 1, True) < 1, 0., 1., )  # all view==1, mask=1
        disp = torch.cat(disp, 1)
        disp_mean = torch.mean(disp, 1, True)
        return disp_mean, raw_warp_img

    def warp(self, disp, views_list, idx):
        B, C, H, W = views_list[0].shape
        x, y = torch.arange(0, H), torch.arange(0, W)
        self.meshgrid = torch.stack(torch.meshgrid(x, y), -1).unsqueeze(0)  # 1*H*W*2
        disp = disp.squeeze(1)
        # assert H==self.patch_size and W==self.patch_size,"size is different!"
        tmp = []
        meshgrid = self.meshgrid.repeat(B, 1, 1, 1).to(disp)  # B*H*W*2
        for k in range(9):
            u, v = divmod(idx[k], 9)
            grid = torch.stack([
                torch.clip(meshgrid[:, :, :, 1] - disp * (v - 4), 0, W - 1),
                torch.clip(meshgrid[:, :, :, 0] - disp * (u - 4), 0, H - 1)
            ], -1) / (W - 1) * 2 - 1  # B*H*W*2  归一化到-1，1
            tmp.append(grid_sample(views_list[k], grid, align_corners=True))
        return tmp

    def disparitygression(self, input):
        disparity_values = torch.linspace(-4, 4, 9).to(input)
        x = disparity_values.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        out = torch.sum(torch.multiply(input, x), 1)
        return out.unsqueeze(1)

    def cal_occlusion(self, views):
        views = torch.stack(views, -1)  # B*C*H*W*9
        if self.grad_v == 9:
            grad = torch.mean(torch.abs(views[:, :, :, :, 1:] - views[:, :, :, :, :8]), dim=[1, 4])
        elif self.grad_v == 5:
            grad = torch.mean(torch.abs(views[:, :, :, :, 3:7] - views[:, :, :, :, 2:6]), dim=[1, 4])  # B*H*W
        return grad.unsqueeze(1)


# finetune 0
# views 18
# cycle 0
# transformer 0
class Unsup26_21_56(nn.Module):  # v3 轻量版 final
    def __init__(self, opt, device, is_train=True):
        super(Unsup26_21_56, self).__init__()
        self.is_train = is_train
        self.n_angle = 2
        self.device = device
        self.use_v = opt.use_views
        self.grad_v = opt.grad_v
        self.feat_extract = Feature(in_channels=opt.input_c, out_channels=8)
        feats = 64
        # stack [b*8*9*h*w]*4
        # self.transformer = transformer_layer(32,self.use_v,2,4) 

        self.block3d = nn.ModuleList()
        self.fuse3d = nn.ModuleList()
        for j in range(2):
            '''
            self.block3d.append(nn.Sequential(nn.Conv3d(32, feats, (self.use_v,3,3), 1, padding=(0, 1,1)),nn.LeakyReLU(0.2, True)))
            self.fuse3d.append(nn.Sequential(nn.Conv2d(feats, feats//2, 1, 1),nn.LeakyReLU(0.2, True)))
            '''
            self.block3d.append(
                nn.Sequential(nn.Conv3d(8, feats, (self.use_v, 3, 3), 1, padding=(0, 1, 1)), nn.LeakyReLU(0.2, True)))
            self.fuse3d.append(nn.Sequential(nn.Conv2d(feats, feats, 1, 1), nn.LeakyReLU(0.2, True)))

        feats *= 2
        self.fuse2d = nn.ModuleList()
        for j in range(4):
            self.fuse2d.append(nn.Sequential(nn.Conv2d(feats, feats, 3, 1, padding=1), nn.LeakyReLU(0.2, True),
                                             nn.Conv2d(feats, feats, 1, 1)))
            self.fuse2d.append(nn.LeakyReLU(0.2, True))

        self.fuse3 = nn.Conv2d(feats, 9, 3, 1, padding=1)
        self.relu3 = nn.Softmax(dim=1)

        # finetune
        # self.warpping = Warpping('train')
        # self.finetune = ContextAdjustmentLayer()

        # warpping

        self.patch_size = opt.input_size

        self.center_index = self.use_v // 2

        # self.upsample = nn.Upsample(scale_factor=2)
        # if self.is_train:
        #     self.base_loss = torch.nn.L1Loss()

    def forward(self, x):

        B, C, H, W = x[0].shape

        feats = []
        feat = self.feat_extract(torch.cat(x[:18], 0))

        feat = adaptive_avg_pool2d(feat, (1, 1)) * feat
        for i in range(18):
            feats.append(feat[i * B:(i + 1) * B, :, :, :])

        feats_angle = []
        for j in range(2):
            feats_tmp = torch.stack(feats[self.use_v * j:self.use_v * (j + 1)], dim=2)  # B*8*9*h*w
            # feats_tmp = self.transformer(feats_tmp)
            feats_tmp = self.block3d[j](feats_tmp)
            feats_tmp = torch.squeeze(feats_tmp, 2)  # B*64*h*w
            feats_angle.append(self.fuse3d[j](feats_tmp))
        cv = torch.cat(feats_angle, 1)
        for j in range(4):
            cv = self.fuse2d[j * 2 + 1](self.fuse2d[j * 2](cv) + cv)

        prob = self.relu3(self.fuse3(cv))
        disp_raw = self.disparitygression(prob)

        return disp_raw, None

    def warp(self, disp, views_list, idx):
        B, C, H, W = views_list[0].shape
        x, y = torch.arange(0, H), torch.arange(0, W)
        self.meshgrid = torch.stack(torch.meshgrid(x, y), -1).unsqueeze(0)  # 1*H*W*2
        disp = disp.squeeze(1)
        # assert H==self.patch_size and W==self.patch_size,"size is different!"
        tmp = []
        meshgrid = self.meshgrid.repeat(B, 1, 1, 1).to(disp)  # B*H*W*2
        for k in range(9):
            u, v = divmod(idx[k], 9)
            grid = torch.stack([
                torch.clip(meshgrid[:, :, :, 1] - disp * (v - 4), 0, W - 1),
                torch.clip(meshgrid[:, :, :, 0] - disp * (u - 4), 0, H - 1)
            ], -1) / (W - 1) * 2 - 1  # B*H*W*2  归一化到-1，1
            tmp.append(grid_sample(views_list[k], grid, align_corners=True))
        return tmp

    def disparitygression(self, input):
        disparity_values = torch.linspace(-4, 4, 9, device=self.device)
        x = disparity_values.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        out = torch.sum(torch.multiply(input, x), 1)
        return out.unsqueeze(1)

    def cal_occlusion(self, views):
        views = torch.stack(views, -1)  # B*C*H*W*9
        if self.grad_v == 9:
            grad = torch.mean(torch.abs(views[:, :, :, :, 1:] - views[:, :, :, :, :8]), dim=[1, 4])
        elif self.grad_v == 5:
            grad = torch.mean(torch.abs(views[:, :, :, :, 3:7] - views[:, :, :, :, 2:6]), dim=[1, 4])  # B*H*W
        return grad.unsqueeze(1)


# finetune 0
# views 36
# cycle 1
# transformer 0
class Unsup22_13_01(nn.Module):  # v3
    def __init__(self, opt, device, is_train=True):
        super(Unsup22_13_01, self).__init__()
        self.is_train = is_train
        self.n_angle = 2
        feats = 32
        self.device = device
        self.use_v = opt.use_views
        self.grad_v = opt.grad_v
        self.feat_extract = Feature(in_channels=opt.input_c, out_channels=8)

        self.block3d = nn.ModuleList()
        self.fuse3d = nn.ModuleList()
        for j in range(4):
            self.block3d.append(
                nn.Sequential(nn.Conv3d(8, 64, (self.use_v, 3, 3), 1, padding=(0, 1, 1)), nn.LeakyReLU(0.2, True)))
            self.fuse3d.append(nn.Sequential(nn.Conv2d(64, 32, 1, 1), nn.LeakyReLU(0.2, True)))

        feats *= 4
        self.fuse2d = nn.ModuleList()
        for j in range(4):
            self.fuse2d.append(nn.Sequential(nn.Conv2d(feats, feats, 3, 1, padding=1), nn.LeakyReLU(0.2, True),
                                             nn.Conv2d(feats, feats, 1, 1)))
            self.fuse2d.append(nn.LeakyReLU(0.2, True))

        self.fuse3 = nn.Conv2d(feats, 9, 3, 1, padding=1)
        self.relu3 = nn.Softmax(dim=1)
        self.transformer = transformer_layer(32, self.use_v, 2, 4)
        self.finetune = ContextAdjustmentLayer()

    def forward(self, x):
        feats = []

        for xi in x:
            feat_i = self.feat_extract(xi)
            atten_i = adaptive_avg_pool2d(feat_i, (1, 1))
            feats.append(feat_i * atten_i)

        feats_angle = []
        for j in range(4):
            feats_tmp = torch.stack(feats[self.use_v * j:self.use_v * (j + 1)], dim=2)  # B*8*9*h*w
            # feats_tmp = self.transformer(feats_tmp)
            feats_tmp = self.block3d[j](feats_tmp)
            feats_tmp = torch.squeeze(feats_tmp, 2)  # B*64*h*w
            feats_angle.append(self.fuse3d[j](feats_tmp))

        cv = torch.cat(feats_angle, 1)
        for j in range(4):
            cv = self.fuse2d[j * 2 + 1](self.fuse2d[j * 2](cv) + cv)

        prob = self.relu3(self.fuse3(cv))
        disp_raw = self.disparitygression(prob)

        return disp_raw, None

    def warp(self, disp, views_list, idx):
        B, C, H, W = views_list[0].shape
        x, y = torch.arange(0, H), torch.arange(0, W)
        self.meshgrid = torch.stack(torch.meshgrid(x, y), -1).unsqueeze(0)  # 1*H*W*2
        disp = disp.squeeze(1)
        # assert H==self.patch_size and W==self.patch_size,"size is different!"
        tmp = []
        meshgrid = self.meshgrid.repeat(B, 1, 1, 1).to(disp)  # B*H*W*2
        for k in range(9):
            u, v = divmod(idx[k], 9)
            grid = torch.stack([
                torch.clip(meshgrid[:, :, :, 1] - disp * (v - 4), 0, W - 1),
                torch.clip(meshgrid[:, :, :, 0] - disp * (u - 4), 0, H - 1)
            ], -1) / (W - 1) * 2 - 1  # B*H*W*2  归一化到-1，1
            tmp.append(grid_sample(views_list[k], grid, align_corners=True))
        return tmp

    def disparitygression(self, input):
        disparity_values = torch.linspace(-4, 4, 9, device=self.device)
        x = disparity_values.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        out = torch.sum(torch.multiply(input, x), 1)
        return out.unsqueeze(1)

    def cal_occlusion(self, views):
        views = torch.stack(views, -1)  # B*C*H*W*9
        if self.grad_v == 9:
            grad = torch.mean(torch.abs(views[:, :, :, :, 1:] - views[:, :, :, :, :8]), dim=[1, 4])
        elif self.grad_v == 5:
            grad = torch.mean(torch.abs(views[:, :, :, :, 3:7] - views[:, :, :, :, 2:6]), dim=[1, 4])  # B*H*W
        return grad.unsqueeze(1)



'''
真实场景
'''
# class OPENetmodel(BaseModel):  # Basemodel
#     @staticmethod
#     def modify_commandline_options(parser, is_train=True):
#         return parser
#
#     def __init__(self, opt):
#         BaseModel.__init__(self, opt)
#         self.loss_names = ['L1']
#         if opt.losses.find(
#                 'smooth') != -1:  # 判断 opt.losses 字符串中是否包含子字符串 'smooth'，如果包含则向 self.loss_names 列表中添加一个元素 'smoothness'。
#             self.loss_names.append('smoothness')
#         self.visual_names = ['center_input', 'output', 'label']
#
#         self.model_names = ['EPI']
#         net = eval(self.opt.net_version)(opt, self.device,
#                                          self.isTrain)  # eval(self.opt.net_version)的作用是返回网络版本，即下面的Unsup31_15_53、Unsup27_16_16或Unsup26_21_56
#         # net = Unsup31_15_53(opt, self.device,self.isTrain)   # final syth:07-13-39  09-10-27(增强) real:08-1-12
#         # net = Unsup27_16_16(opt, self.device,self.isTrain) # finetune syth:07-13-42
#         # net = Unsup26_21_56(opt, self.device,self.isTrain) # fast syth:07-13-44  07-20-46  real: 08-01-23
#
#         self.netEPI = init_net(net, opt.init_type, opt.init_gain, self.gpu_ids)
#         self.use_v = opt.use_views  # 表示9*9的子孔径图像，索引为0~8（一行或者一列），use_views为9
#
#         self.center_index = self.use_v // 2  # 将use_views的一半作为中心视图的索引：4
#         self.alpha = opt.alpha
#         self.pad = opt.pad
#         self.lamda = opt.lamda
#
#         self.test_loss_log = 0  # 将初始测试损失值设为 0。
#         self.test_loss = torch.nn.L1Loss()  # 创建一个 L1 损失函数对象
#         if self.isTrain:
#             # define loss functions
#             self.criterionL1 = eval(self.opt.loss_version)(self.opt, self.device)
#             self.optimizer = get_optimizer(opt, filter(lambda p: p.requires_grad, self.netEPI.parameters()), LR=opt.lr)
#             self.optimizers.append(self.optimizer)
#         # 这段代码片段展示了在训练模式下（self.isTrain 为真时）进行的操作：
#         # self.criterionL1 = eval(self.opt.loss_version)(self.opt, self.device): 根据 self.opt.loss_version 的值，动态地创建了一个损失函数对象，并将其赋值给
#         # self.criterionL1。通过 eval() 函数执行字符串表示的表达式，这里假设 self.opt.loss_version 是一个字符串，代表了要使用的损失函数的名称或类型。
#         # self.optimizer = get_optimizer(opt, filter(lambda p: p.requires_grad, self.netEPI.parameters()), LR=opt.lr): 调用 get_optimizer 函数来获取一个优化器对象，
#         # 该函数接受一些参数，包括模型的可训练参数（通过 filter(lambda p: p.requires_grad, self.netEPI.parameters()) 筛选出需要梯度更新的参数）、学习率等参数。最后，将获取的优化器对象赋值给 self.optimizer。
#         # lambda p: p.requires_grad 是一个匿名函数，用于判断参数 p 是否需要计算梯度（即 requires_grad 属性为 True）。self.netEPI.parameters() 返回模型 self.netEPI 的所有参数。
#         # 通过 filter 函数，仅保留那些需要计算梯度的参数，最终返回一个迭代器或列表，其中包含需要梯度更新的参数对象。
#         # self.optimizers.append(self.optimizer): 将创建的优化器对象 self.optimizer 添加到 self.optimizers 列表中，这可能是为了在训练过程中管理多个优化器。
#         # 这些操作通常在模型训练阶段用于设置损失函数和优化器，以便在训练循环中使用它们来计算损失并更新模型参数。
#
#     def set_input(self, inputs, epoch):
#         self.epoch = epoch
#         # self.supervise_view = rearrange(inputs[0].to(self.device), 'b c (h1 h) (w1 w) u v -> (b h1 w1) c h w u v', h1=8, w1=8)
#         self.supervise_view = inputs[0].to(
#             self.device)  # 这段代码将输入数据 inputs[0] 移动（或复制）到指定的设备上，即将数据传输到 GPU 或 CPU 上进行处理。具体来说，inputs[0] 是模型的输入数据，通过调用 .to(self.device) 方法，将其移动到在类中定义的 self.device 上。
#         # 移动数据到特定设备上的操作通常用于确保模型在训练或推理过程中能够利用 GPU 等加速器来加快计算速度。通过在不同设备上存储和处理数据，可以充分利用硬件资源，提高模型的性能表现。
#         self.input = []
#         for j in range(self.use_v):  # 对 self.supervise_view 数据进行切片操作，选择特定的数据子集。具体来说，self.supervise_view[:,:,:,:, self.center_index,j] 使用了多维切片操作，其中 : 表示选择该维度上的所有元素，self.center_index 和 j 则表示在最后两个维度上选择特定的索引。
#             self.input.append(self.supervise_view[:, :, :, :, self.center_index, j])  # 行
#         for j in range(self.use_v):
#             self.input.append(self.supervise_view[:, :, :, :, j, self.center_index])  # 列
#         for j in range(self.use_v):
#             self.input.append(self.supervise_view[:, :, :, :, j, j])  # 对角线
#         for j in range(self.use_v):
#             self.input.append(self.supervise_view[:, :, :, :, j, self.use_v - 1 - j])  # 反对角线（此处应该为self.use_v-j吧）
#         # ------------输入是来自四个不同方向的子孔径图像---------------
#         self.center_input = self.input[self.center_index]
#         self.label = inputs[1].to(self.device)
#         # 这段代码是一个方法 set_input，用于设置模型的输入数据。下面是这个方法的主要步骤：
#         # 将输入中的第一个元素 inputs[0] 移动到设备（如 GPU）上，然后赋值给 self.supervise_view。注释部分的代码使用了 rearrange 函数对输入进行了一些重排操作，但实际上并没有被使用。
#         # 初始化空列表 self.input 用来存储处理后的输入数据。
#         # 通过循环，从 self.supervise_view 中提取特定位置的数据，并将其添加到 self.input 列表中。具体来说，根据 center_index 和 use_v 的值，从 self.supervise_view 中选择不同位置的数据添加到 self.input 中。
#         # 然后，将 self.input 中的第 center_index 个元素赋值给 self.center_input。
#         # 最后，将输入中的第二个元素 inputs[1] 移动到设备上，并赋值给 self.label。
#         # 这个方法的作用是根据输入数据中的特定位置信息，构建模型的输入数据，为模型提供正确的数据格式以及需要关注的信息，同时准备好模型的标签数据。这样有助于模型在训练和推理过程中正确地处理输入数据。
#
#     # self.netEPI(self.input) 表示将输入数据 self.input 传递给网络模型 self.netEPI 进行计算，得到的结果赋值给 self.output 和 self.raw_warp_img。
#     # self.output 是模型的输出结果，self.raw_warp_img 是网络模型中间的原始扭曲图像。
#     def forward(self):  #############################################################################
#         """Run forward pass; called by both functions <optimize_parameters> and <test>."""
#         self.output, self.raw_warp_img = self.netEPI(self.input)  # G(A)
#         self.test_loss_log = 0
#
#     def backward_G(self):
#         # if self.epoch <= self.opt.n_epochs:
#         #     self.loss_L1 = self.criterionL1(self.output, self.input[:9], self.input[9:18])
#         # else:
#         #     self.loss_L1 = self.criterionL1_5(self.output, self.input[:9], self.input[9:18])
#         # self.loss_L1 = self.criterionL1(self.output, self.input[:9], self.input[9:18])
#         self.loss_L1 = self.criterionL1(self.epoch, self.output, self.input[:9],
#                                         self.input[9:18])  # OPENet-fast，self.input[:9], self.input[9:18]分别代表水平喝垂直方向
#         # 通常 L1 损失是指 Mean Absolute Error（MAE），用于衡量模型输出与目标数据之间的绝对差异。
#         # self.epoch 可能是当前训练的 epoch 数，用于在损失函数中引入一些动态调整或变化；self.output 是模型的输出结果；
#         # 该行代码的作用是调用损失函数 self.criterionL1，并将相关参数传递给该函数，计算得到 L1 损失值。
#         if self.raw_warp_img is None:
#             self.loss_total = self.loss_L1  # 此处的raw_warp_img指什么
#         else:
#             self.loss_raw = 0
#             for views in self.raw_warp_img:
#                 self.loss_raw += self.criterionL1(self.epoch, views)
#             self.loss_total = 0.6 * self.loss_L1 + 0.4 * self.loss_raw  # 从此处可以看出loss_L1是论文中raw损失，loss_raw是final损失
#             # self.loss_total = self.loss_L1
#         self.loss_smoothness = get_smooth_loss(self.output, self.center_input, self.lamda)
#
#         if 'smoothness' in self.loss_names and self.epoch > 2 * self.opt.n_epochs:
#             self.loss_total += self.loss_smoothness
#         self.loss_total.backward()
#         # 这段代码实现了模型的反向传播 backward_G()。该方法用于计算模型参数的梯度，并更新参数。
#         # 具体来说，第一步是计算损失函数 self.loss_total，其中包括两部分：L1 损失 self.loss_L1 和原始扭曲图像损失 self.loss_raw，二者的权重比例为 0.6 和 0.4。
#         # L1 损失 self.loss_L1 衡量了模型输出结果与真实数据之间的差异；原始扭曲图像损失 self.loss_raw 衡量了网络模型中间产生的误差。同时还计算了平滑性损失 self.loss_smoothness，该损失用于缓解输出结果的过度平滑现象。
#         # 接下来，将 self.loss_total 反向传播，并计算参数的梯度。反向传播可以自动计算参数的梯度，并存储在模型中。在优化过程中，使用梯度下降等方法基于梯度更新模型的参数，以优化损失函数值并提升模型的性能表现。
#
#     def optimize_parameters(self):
#         self.netEPI.train()
#         self.forward()
#         self.optimizer.zero_grad()  # 将优化器中所有参数的梯度清零，以准备接收新一轮的梯度计算。
#         self.backward_G()
#         self.optimizer.step()  # 根据优化器中存储的参数梯度信息，更新模型的参数。
#
#
# class Unsup31_15_53(nn.Module):
#     def __init__(self, opt, device, is_train=True):
#         super().__init__()
#         self.is_train = is_train
#         self.n_angle = 2
#         feats = 64
#         self.device = device
#         self.use_v = opt.use_views
#         self.grad_v = opt.grad_v
#         self.patch_size = opt.input_size
#         self.center_index = self.use_v // 2
#         # feature
#         self.feat_extract = Feature(in_channels=opt.input_c, out_channels=8)
#         # transformer
#         ## v3 空间到特征
#         self.transformer = nn.ModuleList()
#         self.block3d = nn.ModuleList()
#         self.fuse3d = nn.ModuleList()
#         for j in range(4):
#             self.transformer.append(transformer_layer(32, 9, 2, 4))  # v2:36 v3:9
#             self.block3d.append(
#                 nn.Sequential(nn.Conv3d(8, 64, (self.use_v, 3, 3), 1, padding=(0, 1, 1)), nn.LeakyReLU(0.2, True)))
#             self.fuse3d.append(nn.Sequential(nn.Conv2d(64, 32, 1, 1), nn.LeakyReLU(0.2, True)))
#             # regression
#         feats = 128
#         self.fuse2d = nn.ModuleList()
#         for j in range(4):
#             self.fuse2d.append(nn.Sequential(nn.Conv2d(feats, feats, 3, 1, padding=1), nn.LeakyReLU(0.2, True),
#                                              nn.Conv2d(feats, feats, 1, 1)))
#             self.fuse2d.append(nn.LeakyReLU(0.2, True))
#         self.fuse3 = nn.Conv2d(feats, 9, 3, 1, padding=1)
#         self.relu3 = nn.Softmax(dim=1)
#         # finetune
#         self.finetune = ContextAdjustmentLayer()
#
#     def forward(self, x):
#         feats = []
#         # feature
#         for xi in x:
#             feat_i = self.feat_extract(xi)
#             # atten_i = adaptive_avg_pool2d(feat_i, (1,1))
#             # feats.append(feat_i*atten_i)
#             feats.append(feat_i)
#
#         # transformer
#         ## v3 10-31-15-53 分别trans24 分别fuse feats：8-64-32 空间到feats
#         feats_angle = []
#         for j in range(4):
#             feats_tmp = torch.stack(feats[self.use_v * j:self.use_v * (j + 1)], 2)
#             feats_tmp = rearrange(feats_tmp, 'b c a (h1 h) (w1 w) -> b (c h1 w1) a h w', h1=2,
#                                   w1=2)  # B*32*9*(h/2)*(w/2)
#             feats_tmp = self.transformer[j](feats_tmp)
#             feats_tmp = rearrange(feats_tmp, 'b (c h1 w1) a h w -> b c a (h1 h) (w1 w)', h1=2,
#                                   w1=2)  # B*8*36*(h/2)*(w/2)
#             feats_tmp = self.block3d[j](feats_tmp)  # B*32*36*(h/2)*(w/2)
#             feats_tmp = torch.squeeze(feats_tmp, 2)
#             feats_angle.append(self.fuse3d[j](feats_tmp))
#
#         # regression
#         cv = torch.cat(feats_angle, 1)  # b,128,h,w
#         for j in range(4):
#             cv = self.fuse2d[j * 2 + 1](self.fuse2d[j * 2](cv) + cv)
#         prob = self.relu3(self.fuse3(cv))
#         disp_raw = self.disparitygression(prob)
#         # finetune
#         occolusion = []
#         disp = []
#         mask = []
#         raw_warp_img = []
#         for j in range(self.n_angle):
#             warpped_views = self.warp(disp_raw, x[self.use_v * j:self.use_v * (j + 1)], IDX[j])
#             if self.is_train:
#                 raw_warp_img.append(warpped_views)
#             occu = self.cal_occlusion(warpped_views)
#             # self.vis(occu, j)
#
#             disp_final, occu_final = self.finetune(disp_raw, occu, x[4])
#             mask.append(torch.where(disp_final < 0.03, 1, 0).float())
#             disp.append(disp_final)
#             # occolusion.append(occu_final)
#         mask = torch.where(torch.mean(torch.cat(mask, 1), 1, True) < 1, 0., 1., )  # all view==1, mask=1
#         disp = torch.cat(disp, 1)
#         disp_mean = torch.mean(disp, 1, True)
#         return disp_mean, raw_warp_img
#
#     def warp(self, disp, views_list, idx):
#         B, C, H, W = views_list[0].shape
#         x, y = torch.arange(0, H), torch.arange(0, W)
#         self.meshgrid = torch.stack(torch.meshgrid(x, y), -1).unsqueeze(0)  # 1*H*W*2
#         disp = disp.squeeze(1)
#         # assert H==self.patch_size and W==self.patch_size,"size is different!"
#         tmp = []
#         meshgrid = self.meshgrid.repeat(B, 1, 1, 1).to(disp)  # B*H*W*2
#         for k in range(9):
#             u, v = divmod(idx[k], 9)
#             grid = torch.stack([
#                 torch.clip(meshgrid[:, :, :, 1] - disp * (v - 4), 0, W - 1),
#                 torch.clip(meshgrid[:, :, :, 0] - disp * (u - 4), 0, H - 1)
#             ], -1) / (W - 1) * 2 - 1  # B*H*W*2  归一化到-1，1
#             tmp.append(grid_sample(views_list[k], grid, align_corners=True))
#         return tmp
#
#     def disparitygression(self, input):
#         disparity_values = torch.linspace(-4, 4, 9).to(input)
#         x = disparity_values.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
#         out = torch.sum(torch.multiply(input, x), 1)
#         return out.unsqueeze(1)
#
#     def cal_occlusion(self, views):
#         views = torch.stack(views, -1)  # B*C*H*W*9
#         if self.grad_v == 9:
#             grad = torch.mean(torch.abs(views[:, :, :, :, 1:] - views[:, :, :, :, :8]), dim=[1, 4])
#         elif self.grad_v == 5:
#             grad = torch.mean(torch.abs(views[:, :, :, :, 3:7] - views[:, :, :, :, 2:6]), dim=[1, 4])  # B*H*W
#         return grad.unsqueeze(1)
