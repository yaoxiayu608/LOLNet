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

IDX = [[36 + j for j in range(9)],
       [4 + 9 * j for j in range(9)],
       [10 * j for j in range(9)],
       [8 * (j + 1) for j in range(9)]]

class Feature(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Feature, self).__init__()

        self.conv1 = nn.Sequential(QuantConv2d(in_channels, out_channels, 3, 1,1,bias=False,input_quant=Int8ActPerTensorFloat), QuantReLU(act_quant=None))

    def forward(self, x):
        x = self.conv1(x)

        return x

class CommonQuant(ExtendedInjector):
    bit_width_impl_type = BitWidthImplType.CONST # 量化中的位宽实现类型为常量
    scaling_impl_type = ScalingImplType.CONST # 量化中的缩放因子实现类型为常量。这意味着在量化过程中，缩放因子的值是固定的，而不是根据数据动态调整的。
    restrict_scaling_type = RestrictValueType.FP # 表示缩放因子的限制类型为浮点数。这可能意味着量化过程中使用的缩放因子以浮点数形式表示，而不是整数。
    zero_point_impl = ZeroZeroPoint # 在量化过程中零点的实现方式是固定的 ZeroZeroPoint 类型
    float_to_int_impl_type = FloatToIntImplType.ROUND # 将浮点数转换为整数时使用四舍五入的方法
    scaling_per_output_channel = False
    narrow_range = True
    signed = True

    @value
    def quant_type(bit_width): # 根据位宽 (bit_width) 确定量化类型 (QuantType)
        if bit_width is None:
            return QuantType.FP
        elif bit_width == 1:
            return QuantType.BINARY
        else:
            return QuantType.INT

class CommonWeightQuant(CommonQuant, WeightQuantSolver): # CommonWeightQuant 类继承自 CommonQuant 和 WeightQuantSolver，专注于权重量化，设置了一个固定的缩放常数。
    scaling_per_output_channel = True
    @value
    def scaling_const(module):
        num_pixel = module.weight[0].nelement()
        
        return module.weight.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(num_pixel)

class CommonActQuant(CommonQuant, ActQuantSolver): # CommonActQuant 类继承自 CommonQuant 和 ActQuantSolver，用于激活量化，并定义了激活值的最小和最大范围。
    min_val = -1.0
    max_val = 1.0

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
        
        self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
                
        self.conv = QuantConv2d(
                    input_channels, 
                    output_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=1,
                    input_quant=CommonActQuant,
                    input_bit_width=1)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        return x

class Unsup27_16_16(nn.Module):  # v3
    def __init__(self):
        super().__init__()
        self.feat_extract = Feature(in_channels=3, out_channels=8)
        self.block = nn.Sequential(BinConv2d(288, 2, 3, 1, 1), QuantReLU(act_quant=None,))
        self.fuse = nn.Sequential(BinConv2d(2, 128, 1, 1), QuantReLU(act_quant=None,))
        self.fuse2d = nn.ModuleList()
        for j in range(4):
            self.fuse2d.append(nn.Sequential(BinConv2d(128, 128, 3, 1, padding=1), QuantReLU(act_quant=None),
                                             BinConv2d(128, 128, 1, 1)))
            self.fuse2d.append(QuantReLU(act_quant=None))
        self.fuse3 = BinConv2d(128, 9, 3, 1, padding=1)
        self.relu3 = nn.Softmax(dim=1)

    def forward(self, x):
        input = torch.cat(x, 0) # [36,3,h,w]
        feat = self.feat_extract(input) # [36,8,h,w]
        b, c, h, w = feat.shape
        x_final = feat.reshape(1, 288, h, w)
        feats_tmp = self.block(x_final)
        cv = self.fuse(feats_tmp)
        
        for j in range(4):
            cv = self.fuse2d[j * 2 + 1](self.fuse2d[j * 2](cv) + cv)
        prob = self.relu3(self.fuse3(cv))
        disp_raw = self.disparitygression(prob)

        return disp_raw

    def disparitygression(self, input):
        disparity_values = torch.linspace(-4, 4, 9)
        x = disparity_values.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        out = torch.sum(torch.multiply(input, x), 1)
        return out.unsqueeze(1)
