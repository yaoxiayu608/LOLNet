import torch.nn as nn
import torch
import torch.functional as F
from brevitas.nn import QuantConv2d,QuantReLU
from brevitas.quant.scaled_int import Int8ActPerTensorFloat
'''
OPAL_BNN
'''

# class BinActive(torch.autograd.Function):   # 这段代码定义了一个名为 BinActive 的自定义 PyTorch 自动求导函数
#     '''
#     Binarize the input activations and calculate the mean across channel dimension.计算缩放因子K并对激活进行二值化
#     '''
#     def forward(self, input):   # 前向传播函数，接收输入 input，并返回二值化后的结果。
#         self.save_for_backward(input)   #  这一行代码保存了输入 input，以便在反向传播时使用。PyTorch 中的自动求导功能需要在前向传播中保存一些中间结果，以便在反向传播时计算梯度。
#         size = input.size()
#         input = input.sign()
#         return input

#     def backward(self, grad_output):    # 输入的梯度 grad_input 就等于输出的梯度 grad_output 乘以 1（如果输入在clip范围内的话）
#         input, = self.saved_tensors     # 从保存的张量中恢复了前向传播时保存的输入 input。
#         grad_input = grad_output.clone()    # 将输出梯度 grad_output 进行克隆，以确保不会影响到原始的梯度张量。
#         grad_input[input.ge(1)] = 0     # 将大于等于1的输入元素对应的梯度置为0。
#         grad_input[input.le(-1)] = 0    # 将小于等于-1的输入元素对应的梯度置为0。
#         return grad_input


# # 自定义 PyTorch 模块，用于实现二值卷积操作。二值卷积操作出现在两个位置，要么用在卷积层，要么用在全连接层，二者都要先进行BN操作，然后对激活进行二值化，最后再加一个relu操作
# class BinConv2d(nn.Module): # change the name of BinConv2d
#     def __init__(self, input_channels, output_channels,
#             kernel_size=-1, stride=-1, padding=0, groups=1, dropout=0,
#             Linear=False, previous_conv=False, size=0):
#         super(BinConv2d, self).__init__()
#         self.input_channels = input_channels
#         self.layer_type = 'BinConv2d'
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dropout_ratio = dropout
#         self.previous_conv = previous_conv

#         if dropout != 0:  # 如果指定了 dropout，就创建一个 nn.Dropout 层，用于在训练过程中随机丢弃输入张量的部分元素，以防止过拟合。
#             self.dropout = nn.Dropout(dropout)
#         self.Linear = Linear    # 根据 Linear 参数的值，判断是否是全连接层。如果不是全连接层，则创建一个 Batch Normalization 层 self.bn 和一个卷积层 self.conv。
#         if not self.Linear:     # Batch Normalization 用于加速训练过程并增强模型的泛化能力，而卷积层则是二维卷积操作。
#             self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)   # 参数 affine=True， 就是设置 对应的两个调节因子 γ 和 β
#             self.conv = nn.Conv2d(input_channels, output_channels,
#                     kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,bias=False)
#         else:
#             if self.previous_conv:  # 如果是全连接层，并且指定了 previous_conv 参数为 True，则创建一个针对输入通道数除以 size 的 Batch Normalization 层 self.bn
#                 self.bn = nn.BatchNorm2d(int(input_channels/size), eps=1e-4, momentum=0.1, affine=True)     # input_channels/size = 50*4*4/4*4=50
#             else:   # 否则创建一个针对输入通道数的 Batch Normalization1D 层。
#                 self.bn = nn.BatchNorm1d(input_channels, eps=1e-4, momentum=0.1, affine=True)
#             self.linear = nn.Linear(input_channels, output_channels)    # 然后，创建一个全连接层 self.linear，用于进行线性变换。

#     def forward(self, x):
#         x = self.bn(x)
#         x = BinActive.apply(x)  # 应用自定义的二值化函数 BinActive.apply() 到输入张量 x 上。
#         if self.dropout_ratio != 0:
#             x = self.dropout(x)
#         if not self.Linear:
#             x = self.conv(x)
#         else:
#             if self.previous_conv:
#                 # print(x.shape)
#                 x = x.view(x.size(0), self.input_channels)  # 这行代码对输入张量 x 进行了形状变换，将其从一个多维张量转换为一个二维张量。
#                 # 将 x 的形状变换为 (batch_size, self.input_channels)，这种形状变换通常在神经网络中用于将卷积层或池化层的输出转换为全连接层的输入，
#                 # 或者将一个多维张量展平为一个二维张量，以便后续的全连接层处理。x.size(0)通常是用于获取张量 x 在其第一个维度上的大小，即张量的批量大小（batch size）。
#             x = self.linear(x)
#         return x



# class conv_block(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride,  downsample=None):
#         super(conv_block, self).__init__()
#         # pad = kernel_size // 2

#         self.downsample = downsample
#         block = nn.ModuleList()
#         block.append(BinConv2d(in_channels, out_channels, 3,1,1))
#         block.append(nn.ReLU(True))
#         block.append(BinConv2d(out_channels, out_channels, 3,1,1))
#         self.conv = nn.Sequential(*block)

#     def forward(self, x):
#         x_skip = x
#         if self.downsample is not None:     # 如果存在 downsample 操作，则对输入 x 进行下采样得到 x_skip。
#             x_skip = self.downsample(x)
#         return self.conv(x) + x_skip    # 对输入 x 进行卷积操作并添加残差连接（residual connection），将其与 x_skip 相加作为最终输出结果。

# class Feature(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3):
#         super(Feature, self).__init__()
#         pad = kernel_size // 2
#         #self.relu = nn.ReLU(True)
#         self.conv1 = nn.Sequential(QuantConv2d(in_channels, 4, 3, 1,1,bias=False,input_quant=Int8ActPerTensorFloat), nn.ReLU(True))

# #         self.layer1 = self._make_layer(4, 4, 2, 1)
# #         self.layer2 = self._make_layer(4, 8, 2, 1)
# #         self.layer3 = self._make_layer(8, 16, 2, 1)


# #         self.branch1 = nn.Sequential(nn.AvgPool2d(2,2), BinConv2d(16, 4,1,1), nn.ReLU(True), nn.UpsamplingBilinear2d(scale_factor=2))   # 双线性上采样层
# #         self.branch2 = nn.Sequential(nn.AvgPool2d(4,4), BinConv2d(16, 4,1,1), nn.ReLU(True), nn.UpsamplingBilinear2d(scale_factor=4))
# #         self.branch3 = nn.Sequential(nn.AvgPool2d(8,8), BinConv2d(16, 4,1,1), nn.ReLU(True), nn.UpsamplingBilinear2d(scale_factor=8))
#         '''
#         此处卷积核大小为3*3的二值卷积注意填充
#         '''
#         self.lastconv = nn.Sequential(BinConv2d(4, 16, 3, 1, 1), nn.ReLU(True), BinConv2d(16,out_channels,1,1))
#         # self.lastconv = nn.Sequential(BinConv2d(28, 16, 3, 1, 1), nn.ReLU(True), BinConv2d(16,out_channels,1,1))

#     def _make_layer(self, in_c, out_c, blocks, stride):
#         downsample = None
#         if stride != 1 or in_c != out_c:    # 步幅不为1或者输入通道数不等于输出通道数就叫做下采样
#             downsample = BinConv2d(in_c, out_c,1,1)   # 创建相应的下采样层 downsample（此处为普通的卷积层）。卷积核的大小为1*1，经过卷积之后特征图大小不变。

#         layers = []
#         layers.append(conv_block(in_c, out_c, 3, stride, downsample))     # 将第一个卷积块添加到 layers 列表中，该卷积块是一个 conv_block 实例。
#         for _ in range(1, blocks):  # 从 1 开始，到 blocks - 1 结束（不包括 blocks）。
#             layers.append(conv_block(out_c, out_c, 3, stride))
#         return nn.Sequential(*layers)   # 最后，将 layers 列表中的层按顺序组合成一个序列模块并返回。

#     def forward(self, x):
#         x = self.conv1(x)   # 第一层卷积保留全精度
# #         l1 = self.layer1(x)
# #         l2 = self.layer2(l1)
# #         l3 = self.layer3(l2)

# #         x = torch.cat([l3, self.branch1(l3), self.branch2(l3), self.branch3(l3)], 1)
#         x = self.lastconv(x)
#         return x

'''
上面是OPAL_BNN
'''

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
        block = []      # 创建了一个空列表 block，用于存储卷积和批归一化层。
        block.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride, pad))
        block.append(nn.BatchNorm3d(out_channels))
        self.conv = nn.Sequential(*block)   # 最后，将整个 block 列表作为参数传递给 nn.Sequential，以创建一个包含卷积和批归一化的序列模块 self.conv。

    def forward(self, x):
        return self.conv(x)

class conv_block(nn.Module):    # dilation 是卷积的膨胀率。downsample 是一个可选参数，用于对输入进行下采样（降采样）的操作。
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
        if self.downsample is not None:     # 如果存在 downsample 操作，则对输入 x 进行下采样得到 x_skip。
            x_skip = self.downsample(x)
        return self.conv(x) + x_skip    # 对输入 x 进行卷积操作并添加残差连接（residual connection），将其与 x_skip 相加作为最终输出结果。

class Feature(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Feature, self).__init__()
        pad = kernel_size // 2
        self.relu = nn.LeakyReLU(0.2, True)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 4, 3, 1,1), nn.LeakyReLU(0.2, True))

        self.layer1 = self._make_layer(4, 4, 2, 1, 1)
        self.layer2 = self._make_layer(4, 8, 2, 1, 1)
        self.layer3 = self._make_layer(8, 16, 2, 1, 1)


        self.branch1 = nn.Sequential(nn.AvgPool2d(2,2), conv_2d(16, 4,1,1), nn.LeakyReLU(0.2, True), nn.UpsamplingBilinear2d(scale_factor=2))   # 双线性上采样层
        self.branch2 = nn.Sequential(nn.AvgPool2d(4,4), conv_2d(16, 4,1,1), nn.LeakyReLU(0.2, True), nn.UpsamplingBilinear2d(scale_factor=4))
        self.branch3 = nn.Sequential(nn.AvgPool2d(8,8), conv_2d(16, 4,1,1), nn.LeakyReLU(0.2, True), nn.UpsamplingBilinear2d(scale_factor=8))

        self.lastconv = nn.Sequential(conv_2d(28, 16, 3, 1), nn.LeakyReLU(0.2, True), nn.Conv2d(16,out_channels,1,1))

    def _make_layer(self, in_c, out_c, blocks, stride, dilation):
        downsample = None
        if stride != 1 or in_c != out_c:
            downsample = conv_2d(in_c, out_c,1,1)   # 创建相应的下采样层 downsample。

        layers = []
        layers.append(conv_block(in_c, out_c, 3, stride, dilation, downsample))     # 将第一个卷积块添加到 layers 列表中，该卷积块是一个 conv_block 实例。
        for _ in range(1, blocks):  # blocks 表示卷积块中的卷积层数量。循环添加额外的卷积块到 layers 列表中，每个卷积块也是一个 conv_block 实例。
            layers.append(conv_block(out_c, out_c, 3, stride, dilation))
        return nn.Sequential(*layers)   # 最后，将 layers 列表中的层按顺序组合成一个序列模块并返回。

    def forward(self, x):
        x = self.conv1(x)
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)

        x = torch.cat([l3, self.branch1(l3), self.branch2(l3), self.branch3(l3)], 1)
        x = self.lastconv(x)
        return x

def default_conv(in_channels, out_channels, kernel_size, bias= False):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


## conv-bn-relu
class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=nn.BatchNorm2d):
        super(ConvBlock, self).__init__()
        padding = kernel_size // 2
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out



## x-conv-bn-conv-bn-relu
##  --------------------^
class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):   # res_scale 是残差比例，默认为 1。

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)    # 激活函数只在第一次循环时添加。

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)  # res_scale 是残差比例，通过 mul(self.res_scale) 将残差结果乘以这个比例。
        res += x
        # 残差比例（residual scaling factor）在残差块中用于调整残差连接的大小，控制着残差部分对输出的贡献程度。在残差块中，残差比例是一个乘法因子，用于缩放通过残差模块产生的残差值，然后将其与输入进行相加。
        return res

## Channel Attention (CA) Layer 用于实现通道注意力机制，通过学习通道之间的重要性权重，有助于模型更好地关注重要的特征通道，提升模型性能和泛化能力。
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):  # reduction 是通道压缩比例，默认为 16。
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)     # 创建了一个全局平均池化层 avg_pool，用于将特征图进行全局平均池化操作，将特征图转换为一个点。
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(   # 创建了一个包含两个卷积层和一个激活函数的序列 conv_du，用于对通道权重进行降维和升维的操作，最终输出通道注意力权重。
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),  # 将输入特征的通道数降低为原来的 channel // reduction。
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),  # 将通道数升高回原来的 channel。
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)  结合了残差连接和通道注意力机制，有助于提升神经网络对特征的提取和利用效果。
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG) 用于实现具有残差连接的残差块组（Residual Group）。
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):  # n_resblocks（残差块数量）
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(   # 这段代码使用了列表推导式来创建一个包含多个 RCAB 实例的列表 modules_body。具体来说，它循环创建了 n_resblocks 个 RCAB 实例，并将它们添加到 modules_body 列表中。
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]    # 这段代码中的反斜杠（\）用于表示换行符的续行符号，它告诉解释器下一行是当前语句的延续。这种技术通常用于在代码很长时，以使代码更易阅读。
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## dense block
## bn-relu-1*1-bn-relu-3*3
class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate=0):
        super(_DenseLayer, self).__init__()
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv2d(num_input_features, bn_size*growth_rate,
                                            kernel_size=1, stride=1, bias=False))
        self.add_module("norm2", nn.BatchNorm2d(bn_size*growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(bn_size*growth_rate, growth_rate,
                                            kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):   # 在 forward 方法中，通过调用 super(_DenseLayer, self).forward(x) 来实现对上述添加的层的前向传播计算。
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Sequential):   # 用于构建 DenseNet 中的稠密块（Dense Block）。
    """DenseBlock
       growth_rate: 增长率， outchannel = num_input_features + (num_layers-1) * growth_rate
    """
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate=0):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):     # 通过一个循环遍历 num_layers 次，每次创建一个 _DenseLayer 类的实例作为稠密块的一层，并将其添加到当前的 _DenseBlock 中。
            layer = _DenseLayer(num_input_features+i*growth_rate, growth_rate, bn_size,
                                drop_rate)
            self.add_module("denselayer%d" % (i+1,), layer)     # "denselayer%d" % (i+1,) 会动态生成类似 "denselayer1", "denselayer2", "denselayer3", ... 的字符串作为层的名称。

class _Transition(nn.Sequential):   # 用于构建 DenseNet 中的过渡层（Transition Layer）。
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        # self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


## OctaveConv 这段代码定义了一个名为 OctaveConv 的自定义模块，用于实现 Octave Convolution，即八分之一卷积。在这个模块中，输入特征被分成高频部分和低频部分，然后分别进行卷积操作，最后将结果合并。
class OctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,  # alpha：分频比例，默认为 0.5
                 groups=1, bias=False):
        super(OctaveConv, self).__init__()
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)  # 平均池化层，池化核大小为 (2, 2)，步长为 2,用于将高频部分转换为低频部分
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')   # 以 2 倍的比例进行上采样，而且在上采样过程中使用最邻近插值的方式来填充新生成的像素值，用于将低频部分上采样至高频部分的尺寸
        self.stride = stride    # l2l、l2h、h2l、h2h：四个卷积层，分别对应低频到低频、低频到高频、高频到低频、高频到高频的卷积操作
        self.l2l = torch.nn.Conv2d(int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.l2h = torch.nn.Conv2d(int(alpha * in_channels), out_channels - int(alpha * out_channels),  # 输入通道数为 int(alpha * in_channels)，其中 alpha 是分频比例，控制低频部分的通道数。
                                   kernel_size, 1, padding, dilation, groups, bias)  # 输出通道数为总输出通道数减去低频部分的输出通道数，保证高频部分的输出通道数。
        self.h2l = torch.nn.Conv2d(in_channels - int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels - int(alpha * in_channels),
                                   out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
    # 在前向传播方法 forward 中，首先对输入特征进行分离，然后根据步长的不同对高频部分和低频部分进行池化操作。接下来分别对四个部分进行卷积操作，并通过上采样和相加操作将结果合并，最终返回合并后的高频部分和低频部分。
    def forward(self, x):
        X_h, X_l = x

        if self.stride ==2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_h2l = self.h2g_pool(X_h)

        X_h2h = self.h2h(X_h)
        X_l2h = self.l2h(X_l)

        X_l2l = self.l2l(X_l)
        X_h2l = self.h2l(X_h2l)

        X_l2h = self.upsample(X_l2h)
        X_h = X_l2h + X_h2h
        X_l = X_h2l + X_l2l

        return X_h, X_l

class FirstOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(FirstOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.h2l = torch.nn.Conv2d(in_channels, int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels, out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        if self.stride ==2:
            x = self.h2g_pool(x)

        X_h2l = self.h2g_pool(x)
        X_h = x
        X_h = self.h2h(X_h)
        X_l = self.h2l(X_h2l)

        return X_h, X_l

class LastOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(LastOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2,2), stride=2)

        self.l2h = torch.nn.Conv2d(int(alpha * in_channels), out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels - int(alpha * in_channels),
                                   out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        X_h, X_l = x

        if self.stride ==2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_l2h = self.l2h(X_l)
        X_h2h = self.h2h(X_h)
        X_l2h = self.upsample(X_l2h)

        X_h = X_h2h + X_l2h

        return X_h

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU(0.2, True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConvBn3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


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


class Hourglass3d(nn.Module):   # 这个设计实现了 Hourglass 结构中的特征金字塔和多尺度特征融合，有利于提高模型对输入数据的特征表达能力。
    def __init__(self, channels):   # 在 __init__ 函数中，首先定义了一系列卷积层和反卷积层
        super(Hourglass3d, self).__init__()

        self.conv1a = ConvBnReLU3D(channels, channels * 2, kernel_size=3, stride=2, pad=1)
        self.conv1b = ConvBnReLU3D(channels * 2, channels * 2, kernel_size=3, stride=1, pad=1)

        self.conv2a = ConvBnReLU3D(channels * 2, channels * 4, kernel_size=3, stride=2, pad=1)
        self.conv2b = ConvBnReLU3D(channels * 4, channels * 4, kernel_size=3, stride=1, pad=1)

        self.dconv2 = nn.Sequential(
            nn.ConvTranspose3d(channels * 4, channels * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels * 2))

        self.dconv1 = nn.Sequential(
            nn.ConvTranspose3d(channels * 2, channels, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels))

        self.redir1 = ConvBn3D(channels, channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = ConvBn3D(channels * 2, channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1b(self.conv1a(x))
        conv2 = self.conv2b(self.conv2a(conv1))
        dconv2 = F.relu(self.dconv2(conv2) + self.redir2(conv1), inplace=True)
        dconv1 = F.relu(self.dconv1(dconv2) + self.redir1(x), inplace=True)
        return dconv1


def homo_warping(src_fea, src_proj, ref_proj, depth_values):    # 这段代码实现了一个函数 homo_warping，用于对输入的特征图进行透视变换。
    # src_fea: [B, C, H, W] 源特征图
    # src_proj: [B, 4, 4]   源投影矩阵，表示将源特征图从相机坐标系投影到世界坐标系的变换矩阵。
    # ref_proj: [B, 4, 4]   参考投影矩阵，表示将参考特征图从相机坐标系投影到世界坐标系的变换矩阵。
    # depth_values: [B, Ndepth] 深度值，其中 Ndepth 是深度值的数量。
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))  # 根据源投影矩阵 src_proj 和参考投影矩阵 ref_proj，计算旋转矩阵 rot 和平移向量 trans。这里使用了 Torch 的矩阵乘法和求逆操作。
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),  # 根据源特征图的高度和宽度，生成网格坐标 (x, y)，并将其转换为齐次坐标形式 xyz，大小为 [B, 3, H*W]。
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]     将网格坐标 xyz 进行旋转变换得到 rot_xyz，大小为 [B, 3, H*W]。
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,    # 根据深度值和 rot_xyz，计算旋转后的深度坐标 rot_depth_xyz
                                                                                            1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W] 将平移向量 trans 扩展为 [B, 3, 1, 1] 的形状，并与 rot_depth_xyz 相加，得到投影后的三维坐标 proj_xyz
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W] 将 proj_xyz 中的 x、y 坐标除以 z 坐标，得到归一化的投影坐标 proj_xy
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1   # 对归一化的投影坐标进行归一化处理，得到范围在 [-1, 1] 的归一化坐标 proj_x_normalized 和 proj_y_normalized，大小为 [B, Ndepth, H*W]。
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy
    # 使用 F.grid_sample 函数对源特征图 src_fea 进行插值采样，得到经过透视变换后的特征图 warped_src_fea，大小为 [B, C, Ndepth, H, W]。
    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea


from einops import rearrange
import math

"""

input  : B*C*A*H*W

output : B*C*A*H*W

"""

class transformer_layer(nn.Module):
    def __init__(self,channels,angles,layer_num,num_heads):
        super().__init__()
        self.channels = channels
        self.angRes = angles

        self.pos_encoding = PositionEncoding(temperature=10000)
        self.MHSA_params = {}
        self.MHSA_params['num_heads'] = num_heads
        self.MHSA_params['dropout'] = 0.
        # 调用 make_layer 方法创建了一系列交替的角度变换（AngTrans）和空间变换（SpaTrans）层，并将它们组合成一个神经网络层。
        ################ Alternate AngTrans & SpaTrans ################
        self.altblock = self.make_layer(layer_num=layer_num)

    def make_layer(self, layer_num):
        layers = []
        for i in range(layer_num):
            layers.append(AngTrans(self.channels,self.angRes,self.MHSA_params))
        return nn.Sequential(*layers)

    def forward(self, lr):
        # 前向传播函数，接受输入 lr，并对其进行处理。在前向传播过程中，首先根据输入的尺寸调整每个子模块（AngTrans 层）的高度和宽度属性。
        # [B, C(hannels), A, h, w]
        for m in self.modules():
            m.h = lr.size(-2)
            m.w = lr.size(-1)

        buffer = lr

        # Position Encoding
        ang_position = self.pos_encoding(buffer, dim=2, token_dim=self.channels)
        for m in self.modules():
            m.ang_position = ang_position

        # Alternate AngTrans & SpaTrans
        out = self.altblock(buffer) + buffer

        return out


class PositionEncoding(nn.Module):
    def __init__(self, temperature):
        super(PositionEncoding, self).__init__()
        self.temperature = temperature

    def forward(self, x, dim, token_dim):
        self.token_dim = token_dim # 8
        grid_dim = torch.linspace(0, self.token_dim - 1, self.token_dim, dtype=torch.float32)
        grid_dim = 2 * (grid_dim // 2) / self.token_dim
        grid_dim = self.temperature ** grid_dim

        pos_size = [1, 1, 1, 1, 1, self.token_dim]
        length = x.size(dim)
        pos_size[dim] = length

        pos_dim = (torch.linspace(0, length - 1, length, dtype=torch.float32).view(-1, 1) / grid_dim).to(x.device)
        pos_dim = torch.cat([pos_dim[:, 0::2].sin(), pos_dim[:, 1::2].cos()], dim=1)
        pos_dim = pos_dim.view(pos_size)

        position = pos_dim

        position = rearrange(position, 'b 1 a h w dim -> b dim a h w')

        return position


class AngTrans(nn.Module):
    def __init__(self, channels, angRes, MHSA_params):
        super(AngTrans, self).__init__()
        self.angRes = angRes #9
        self.ang_dim = channels #8
        self.norm = nn.LayerNorm(self.ang_dim) #8
        self.attention = nn.MultiheadAttention(self.ang_dim,
                                               MHSA_params['num_heads'],
                                               MHSA_params['dropout'],
                                               bias=False)
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.attention.out_proj.bias = None

        self.feed_forward = nn.Sequential(
            nn.LayerNorm(self.ang_dim),
            nn.Linear(self.ang_dim, self.ang_dim * 2, bias=False),
            nn.ReLU(True),
            nn.Dropout(MHSA_params['dropout']),
            nn.Linear(self.ang_dim * 2, self.ang_dim, bias=False),
            nn.Dropout(MHSA_params['dropout'])
        )

    @staticmethod
    def SAI2Token(buffer):

        buffer_token = rearrange(buffer, 'b c a h w -> a (b h w) c')
        return buffer_token

    def Token2SAI(self, buffer_token):
        buffer = rearrange(buffer_token, '(a) (b h w) (c) -> b c a h w', a=self.angRes, h=self.h, w=self.w)
        return buffer

    def forward(self, buffer):
        ang_token = self.SAI2Token(buffer)
        ang_PE = self.SAI2Token(self.ang_position)
        ang_token_norm = self.norm(ang_token + ang_PE)

        ang_token = self.attention(query=ang_token_norm,
                                   key=ang_token_norm,
                                   value=ang_token,
                                   need_weights=False)[0] + ang_token

        ang_token = self.feed_forward(ang_token) + ang_token
        buffer = self.Token2SAI(ang_token)

        return buffer

if __name__ == '__main__':

    model = transformer_layer(8,9,1,4)
    x = torch.randn((4,8,9,64,64))
    y = model(x)
    print()
