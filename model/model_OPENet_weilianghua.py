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
from BinOp import BinOp
from .loss import get_smooth_loss, noOPAL, Lossv4, get_distillation_loss
from .basemodel import BaseModel, init_net, get_optimizer
from .layers import *
from .context_adjustment_layer import *
import time
import os
import brevitas.nn as qnn
from brevitas.nn import QuantConv2d, QuantIdentity, QuantLinear, QuantReLU
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

class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    def forward(self, input):
        self.save_for_backward(input)
        size = input.size()
        input = input.sign()
        return input

    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class BinConv2d(nn.Module): # change the name of BinConv2d
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=-1, padding=-1, groups=1, dropout=0,
            Linear=False):
        super(BinConv2d, self).__init__()
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout


        self.bn = nn.BatchNorm2d(output_channels, eps=1e-4, momentum=0.1, affine=True)
        self.conv = nn.Conv2d(input_channels, output_channels,
                kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        
        x = BinActive.apply(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

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

#         self.conv = QuantConv2d(
#                     in_channels=in_channels,
#                     out_channels=out_channels,
#                     kernel_size=kernel_size,
#                     stride = stride,
#                     dilation = dilation,
#                     padding = padding,
#                     groups = groups,
#                     bias=False,
#                     weight_quant=weight_quant,
#                     weight_bit_width=weight_bit_width)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = QuantReLU(
#                                 act_quant=act_quant,
#                                 bit_width=act_bit_width,
#                                 per_channel_broadcastable_shape=(1, out_channels, 1, 1),
#                                 scaling_per_channel=False)
#         # self.dropout = nn.Dropout(p=0.05)  # p表示随机丢弃对应比列的神经元
#     def forward(self, input):
#         output = self.conv(input)
#         output = self.bn(output)
#         output = self.relu(output) 
#         # output = self.dropout(output) 
#         return output


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

        self.teacher_model = TeacherModel(self.device)
        save_dir = os.path.join('./checkpoints', 'OPENet', '2025-02-20-06-33')
        load_filename = '%s_net_%s.pth' % ('282', 'EPI')
        load_path = os.path.join(save_dir,load_filename)
        self.teacher_model.load_state_dict(torch.load(load_path, map_location=str(self.device)))  # 加载权重文件
        self.teacher_model.to(self.device)

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
        self.output, self.raw_warp_img = self.netEPI(self.gray_input)  # G(A)
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
        self.loss_L1 = self.criterionL1(self.epoch,self.output, self.input[:9], self.input[9:18], self.input[18:27],self.input[27:])
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
        self.loss_distillation = get_distillation_loss(self.output, self.teacher_outputs, self.epoch)
        # 蒸馏损失比光度一致性损失小一个数量级
        # print('loss_total',self.loss_total)
        # print('loss_distillation',self.loss_distillation)
        self.loss_total = self.loss_total + self.loss_distillation
        # self.loss_total += self.loss_distillation 
        # 该损失与学习率衰减周期n_epochs有关，但是无论n_epochs设置为多少，一旦加入平滑损失，低比特网络的效果立即变得糟糕
        # if 'smoothness' in self.loss_names and self.epoch > 2*self.opt.n_epochs:
        #     self.loss_total += self.loss_smoothness 
        self.loss_total.backward()
        
    def optimize_parameters(self):
        """
        首先定义二值化操作符
        """
        bin_op = BinOp(self.netEPI)
        self.netEPI.train()
        '''
        模型开始运转之前先对权重进行二值化处理
        '''
        bin_op.binarization()
        self.forward(isTrain=True)
        self.optimizer.zero_grad()  # 将优化器中所有参数的梯度清零，以准备接收新一轮的梯度计算。
        self.backward_G()
        '''
        根据loss.backward()得到梯度之后先进行权重的恢复然后计算二值化权重的梯度
        '''
        bin_op.restore()
        bin_op.updateBinaryGradWeight()
        self.optimizer.step()  # 根据优化器中存储的参数梯度信息，更新模型的参数。         


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


class Unsup27_16_16(nn.Module):  # v3
    def __init__(self,opt,device, is_train=True):
        super().__init__() 
        self.is_train = is_train
        self.n_angle = 2
        feats = 64
        self.device = device
        self.use_v = opt.use_views
        self.grad_v = opt.grad_v
        self.weight_bit_width = 4
        self.act_bit_width = 4
        
        # self.conv1 = ConvBnReLU(28, 64, kernel_size=3, stride=1, pad=1)
        # self.conv2 = ConvBnReLU(64, 64, kernel_size=3, stride=1, pad=1)
        # self.conv3 = ConvBnReLU(64, 64, kernel_size=3, stride=1, pad=1)
        # self.conv6 = ConvBnReLU(64, 64, kernel_size=3, stride=1, pad=1)
        # self.conv7 = ConvBnReLU(64, 50, kernel_size=3, stride=1, pad=1)
        # self.conv4 = nn.Conv2d(50, 9, 3, stride=1, padding=1, bias=False)

        self.conv1 = ConvBnReLU(28, 64, kernel_size=3, stride=1, pad=1)
        self.conv2 = BinConv2d(64, 64, 3, 1, 1)
        self.conv3 = BinConv2d(64, 64, 3, 1, 1)
        self.conv6 = BinConv2d(64, 64, 3, 1, 1)
        self.conv7 = BinConv2d(64, 50, 3, 1, 1)
        self.conv4 = nn.Conv2d(50, 9, 3, stride=1, padding=1, bias=False)

        self.relu3 = nn.Softmax(dim=1)
        
    def forward(self, x):
        
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
        # crop_size = 64
        # _, _, h, w = image.shape
        # # 确保图像尺寸足够大以进行裁剪
        # if h < crop_size or w < crop_size:
        #     raise ValueError("Image size is too small for cropping.")
        # # 随机选择起始点
        # # start_x = np.random.randint(0, w - crop_size)
        # # start_y = np.random.randint(0, h - crop_size)
        # start_x = 200
        # start_y = 400
        # # 裁剪图像
        # image = image[:,:,start_y:start_y+crop_size, start_x:start_x+crop_size]
        # print('final dtype:',image.dtype)  # torch.uint8
        # print('final shape',image.shape)  # [1, 108, 64, 64]
        # np.save("input64.npy", image.permute(0, 2, 3, 1).cpu())
        # cv = self.conv1(image / 255.)
        # cv = cv / 255.  # 这个除以255后面的小数点对可视化效果没有任何影响，从512*512中裁剪出64*64的可视化效果和之前不一致，因为要把64*64当作一个新的输入去处理，这只是映射模式的问题，输出结果的数值大小是一致的？（边界填充会不会造成影响？在计算损失的时候，边缘会不会差距较大，需不需要裁剪掉边缘？）
        # cv = self.conv5(cv)
        # end_time1 = time.time()
        cv = self.conv1(cv)
        cv = self.conv2(cv)
        cv = self.conv3(cv)
        # cv = self.init_feature(cv)
        cv = self.conv6(cv)
        # cv = self.conv8(cv)
        # cv = self.conv9(cv)
        # cv = self.conv10(cv)
        cv = self.conv7(cv)
        # cv = self.conv8(cv)
        # cv = self.conv9(cv)
        # cv = self.conv10(cv)
        # cv = self.conv11(cv)
        # cv = self.conv12(cv)
        cv = self.conv4(cv)  # 注意将conv4的输出通道数设置为9
        # end_time2 = time.time()
        prob = self.relu3(cv)
        disp_raw = self.disparitygression(prob)
        raw_warp_img = []

        return disp_raw, raw_warp_img

    def disparitygression(self, input):
        disparity_values = torch.linspace(-4,4,9,device=self.device)
        x = disparity_values.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        out = torch.sum(torch.multiply(input, x), 1)
        return out.unsqueeze(1)


