# import functools
from abc import ABC, abstractmethod
from torch.optim import lr_scheduler, optimizer
import os
from collections import OrderedDict
import torch
from torch.nn import init


# 抽象基类（Abstract Base Class）
from BinOp import BinOp


class BaseModel(ABC):
    def __init__(self, opt):  # 接收一个参数 opt，该参数应该是一个存储所有实验标志的类的实例，需要是 BaseOptions 的子类。
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt  # 将传入的参数 opt 赋值给类的属性 self.opt，用来存储实验的各种标志和选项。
        self.gpu_ids = opt.gpu_ids  # 将 opt.gpu_ids 赋值给 self.gpu_ids，用来表示可以使用的 GPU 的 ID 列表。
        self.isTrain = opt.isTrain  # 将 opt.isTrain 赋值给 self.isTrain，表示当前是否处于训练模式。
        # 根据 gpu_ids 的值选择使用 CPU 还是 GPU 设备，并将结果赋值给 self.device。
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device(
            'cpu')  # get device name: CPU or GPU
        # 将保存检查点的目录路径设置为 opt.checkpoints_dir/opt.model_name/opt.time_str，其中 opt.checkpoints_dir
        # 是检查点保存的根目录，opt.model_name 是模型名称，opt.time_str 是以当前时间为基础生成的字符串，用于区分不同的训练实验。
        # 通过这个设置，可以在不同的训练实验中保存不同的检查点，便于管理和使用。在进行训练和测试时，模型会将相关结果保存到对应的检查点目录中，以便于后续分析和使用。
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.model_name,
                                     opt.time_str)  # save all the checkpoints to save_dir
        # 如果保存检查点的目录不存在，则创建该目录。
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        # 在数据预处理时使用 scale_width 可能导致输入图像具有不同尺寸，从而影响 cudnn.benchmark 的性能。cudnn 是 NVIDIA 提供的针对深度神经网络的库，它可以优化深度学习模型的计算速度。
        # if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
        # 这行代码是用来启用 cudnn.benchmark 的功能，以优化深度神经网络的计算速度。cudnn.benchmark 可以根据硬件环境动态地选择最优的卷积算法，并且在第一次前向传播时会自动寻找最优的算法，从而加速后续的计算过程。
        # 需要注意的是，启用 cudnn.benchmark 功能时，每次迭代输入数据尺寸不能变化，否则会影响性能。此外，在某些情况下，启用 cudnn.benchmark 可能会导致计算结果不稳定，因此需要根据具体情况进行选择。
        # torch.backends.cudnn.benchmark = True
        #   初始化四个空列表：loss_names、model_names、visual_names 和 optimizers。
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        #   初始化 image_paths 列表和 metric 指标，用于学习率策略中的 'plateau'。
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'

    # 这段代码定义了一个静态方法 modify_commandline_options，用于修改命令行选项解析器的选项。该方法接受两个参数：parser 是原始的选项解析器对象，is_train 是一个布尔值，表示当前是训练阶段还是测试阶段。开发者可以利用这个方法来添加特定于模型的选项，并重写现有选项的默认值。
    # 在方法内部，目前仅简单地将传入的 parser 对象直接返回，即不进行任何修改。开发者可以根据需要在这个方法中添加自定义的模型特定选项，或者根据训练或测试阶段的不同修改选项的默认值。
    # 这种设计模式通常用于为特定模型添加额外的配置选项，以便在命令行中灵活地控制模型的行为。
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    # 这段代码定义了一个抽象方法 set_input，该方法用于从数据加载器中解压缩输入数据并执行必要的预处理步骤。
    # 在方法的文档字符串中，说明了参数 input 是一个字典，其中包含数据本身及其元数据信息。
    # 由于这是一个抽象方法，没有给出具体的实现。子类必须继承这个基类，并根据具体的需求实现 set_input 方法。在子类中，开发者需要编写适当的代码来处理输入数据，例如从数据加载器中获取数据、执行预处理步骤等操作。
    # 这种设计模式常用于模型训练和测试过程中，将数据准备的过程封装在抽象方法中，以确保不同的模型遵循相同的接口规范来设置和准备输入数据。
    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    # 这段代码定义了一个抽象方法 forward，该方法用于执行模型的前向传播过程。文档字符串中说明了该方法会被两个函数调用，分别是 <optimize_parameters> 和 <test>。
    # 由于这是一个抽象方法，没有给出具体的实现。子类必须继承这个基类，并根据具体的需求实现 forward 方法。在子类中，开发者需要编写适当的代码来定义模型的前向传播过程，包括输入数据经过模型各层的计算过程，最终得到输出结果。
    # 这种设计模式常用于定义神经网络模型的接口，确保不同的模型都有统一的前向传播接口，并且可以在训练和测试过程中被调用。
    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    # 这段代码定义了一个抽象方法 optimize_parameters，该方法用于在每个训练迭代中计算损失、梯度并更新网络权重。
    # 由于这是一个抽象方法，没有给出具体的实现。子类必须继承这个基类，并根据具体的需求实现 optimize_parameters 方法。在子类中，开发者需要编写适当的代码来计算损失和梯度，并使用优化器更新网络权重。
    # 这种设计模式常用于定义神经网络模型的训练过程接口，确保不同的模型都有统一的训练接口，并且可以在训练过程中被调用。
    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    # 这段代码定义了一个名为 setup 的方法，用于加载和打印网络，并创建调度器。
    # 在方法中，首先获取参数 opt，然后根据是否处于训练模式来设置调度器。如果是训练模式，会为每个优化器创建一个调度器。接着，尝试加载网络参数，加载的参数根据条件确定具体的后缀。如果加载失败，将输出信息提示从头开始训练。最后，打印网络信息。
    # 需要注意的是，在代码中存在一个异常处理块，它会捕获任何异常并输出相应的信息。这有助于确保即使出现问题，程序也不会崩溃，而是能够继续执行或给出相应的提示信息。
    # 总的来说，这段代码主要用于准备网络训练所需的各项设置，包括加载网络、创建调度器和打印网络信息等操作。
    def setup(self):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        opt = self.opt
        if self.isTrain:
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        # if not self.isTrain:
        try:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
        except Exception:
            print('Train from scratch...')
        self.print_networks(opt.verbose)

    # 这段代码定义了一个 load_networks 方法，用于从磁盘加载所有的网络模型。
    # 在方法中，首先根据传入的参数 epoch 构建要加载的文件名，并依次加载每个网络模型。具体的加载路径由 self.save_dir 和构建的文件名确定。接着，通过 torch.load 方法加载模型参数，并将参数应用到对应的网络模型中。
    # 需要注意的是，在加载模型参数时，对于 PyTorch 版本新于 0.4 的情况（如从 GitHub 源码构建的版本），在 map_location 参数中移除了 str() 方法。另外，在加载参数后，还对包含 InstanceNorm 的检查点进行了修补。
    # 总的来说，这段代码实现了从磁盘加载网络模型参数的功能，以便在训练或推理过程中恢复已保存的模型状态。
    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        # if isinstance(name, str): 的作用是检查变量 name 是否是字符串类型。如果 name 是字符串类型，就会执行 if 语句块内的操作。这样可以确保在处理网络模型名称时，只有当名称是字符串类型时才执行相应的操作，避免出现错误。
        for name in self.model_names:
            if isinstance(name, str):  # isinstance() 是 Python 的一个内置函数，用于检查一个对象是否是一个特定类（class）或类型（type）。
                load_filename = '%s_net_%s.pth' % (epoch, name)  # 根据网络模型名称 name 和当前 epoch 值构建模型参数文件名 load_filename。
                load_path = os.path.join(self.save_dir,
                                         load_filename)  # 通过 os.path.join 方法拼接文件路径 load_path，即参数文件在磁盘中的完整路径。
                # load_path = os.path.join('./checkpoints_0519_block4_base', self.opt.name,load_filename)
                net = getattr(self,
                              'net' + name)  # getattr() 函数从对象中获取指定名称的属性或方法。通过调用 getattr() 函数，可以从 self 对象中获取名为 'net' + name 的属性或方法，并将其存储在变量 net 中。
                # 在 PyTorch 中，torch.nn.DataParallel 是一个用于并行地应用模型到多个 GPU 上的包装器。当模型被放入 torch.nn.DataParallel 中时，它会自动在多个 GPU 上复制模型，并且处理数据的传输等细节。而在使用 torch.nn.DataParallel 包装后，我们需要通过 .module 访问原始的模型实例。
                # 这段代码是在检查 net 是否是 torch.nn.DataParallel 类的实例，如果是，则通过 net.module 获取其内部的模型实例。这样做的目的可能是为了在后续的操作中直接操作原始模型实例，而不是包装后的 DataParallel 实例。
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                # map_location 参数用于将加载的模型参数映射到特定设备上。str(self.device) 用于将设备（device）对象转换为字符串，以便在 map_location 中使用。这样可以确保将模型加载到正确的设备上，即使之前训练和保存模型时使用了不同的设备。
                state_dict = torch.load(load_path, map_location=str(
                    self.device))  # torch.load() 函数读取指定路径下的文件来加载模型的状态字典（state_dict）
                # 在 PyTorch 中，当使用 torch.save() 函数保存模型参数时，会将额外的元数据（metadata）信息保存在 _metadata 属性中。这些元数据可能包含有关模型结构、版本信息或其他辅助信息。
                # 这段代码用于检查 state_dict 中是否存在名为 _metadata 的属性，如果存在的话，则删除这个属性。这样做可能是为了清理掉不必要的元数据信息，以便后续对模型参数进行加载和操作。
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                # 这段代码的作用就是针对旧版本的模型检查点，在加载时对 InstanceNorm 层进行必要的修正。
                # patch InstanceNorm checkpoints prior to 0.4
                # for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                #     self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                # 这行代码用于将加载的模型状态字典 state_dict 加载到神经网络模型 net 中。
                # 在深度学习中，神经网络的参数通常保存在状态字典（state_dict）中。当需要重新加载之前训练好的模型参数时，可以使用 load_state_dict() 方法将状态字典加载到模型中。
                net.load_state_dict(state_dict)

    def print_networks(self, verbose):

        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():  # 通过遍历神经网络模型中的每个参数（net.parameters()），并使用 param.numel() 方法来获取每个参数的元素数量，然后将其累加到 num_params 变量中。
                    num_params += param.numel()  # 在深度学习中，神经网络模型的参数通常由权重和偏置组成，而这些参数又分布在不同的层中。
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (
                    name, num_params / 1e6))  # %s 会被 name 变量的值替换，%f 会被计算出的参数总数量（以百万为单位）替换。
        print(
            '-----------------------------------------------')  # 如果 name 是 'generator'，而 num_params 是 10.5 百万，那么打印的结果将类似于 [Network generator] Total number of parameters : 10.500 M。

    # 这段代码定义了一个名为 print_networks 的函数，用于打印网络模型的总参数数量以及（如果需要的话）网络的架构信息。
    # 该函数接受一个布尔类型的参数 verbose，用于控制是否打印网络的架构信息。当 verbose 为 True 时，会打印网络的架构信息；当 verbose 为 False 时，只会打印网络的总参数数量。
    # 函数首先打印一行提示信息 '---------- Networks initialized -------------'，然后遍历 self.model_names 中的每个模型名称，并获取对应的网络模型。
    # 接下来，对于每个网络模型，函数会计算其总参数数量，并根据 verbose 参数决定是否打印网络的架构信息。最后，函数会打印出每个网络模型的总参数数量，并在末尾打印一行分隔线 '-----------------------------------------------'。
    # 总的来说，这段代码定义了一个用于打印网络模型参数数量和架构信息的函数，方便在训练和调试过程中对网络模型进行查看和分析。

    def set_requires_grad(self, nets, requires_grad=False):

        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:  # 在 Python 中，None 表示空值或者不存在，因此这行代码通过条件判断 net is not None 来确保 net 变量不为空，然后才执行条件块内的操作。
                for param in net.parameters():
                    param.requires_grad = requires_grad

    # 这段代码定义了一个名为 set_requires_grad 的函数，用于设置网络模型参数是否需要梯度计算。
    # 该函数接受两个参数：nets 表示网络模型的列表，requires_grad 是一个布尔值，表示是否需要计算梯度。如果 requires_grad 为 True，则表示需要计算梯度；如果为 False，则表示不需要计算梯度。
    # 函数首先会检查 nets 是否为列表类型，如果不是，则将其转换为列表。接下来，函数会遍历每个网络模型 net，并为每个模型的参数设置 requires_grad 属性为给定的值 requires_grad。
    # 通过将参数的 requires_grad 属性设置为 False，可以避免在反向传播过程中对这些参数进行梯度计算，从而减少不必要的计算量，特别是当我们只需要对部分网络进行训练或微调时，这样的功能非常实用。

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch,
                                                   name)  # 如果 epoch 的值为 10，name 的值为 'generator'，那么经过这行代码处理后，save_filename 的值会变成 10_net_generator.pth。
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)  # net.module 表示从 DataParallel 包装中获取真实的网络模型。
                    net.cuda(self.gpu_ids[0])  # 这行代码用于将网络模型移动到指定的 GPU 上。
                else:
                    torch.save(net.cpu().state_dict(),
                               save_path)  # net.cpu() 的作用是将网络模型移动到 CPU 上，然后调用 state_dict() 方法获取模型的参数字典。然后，torch.save() 函数将获取到的参数字典保存到指定的路径 save_path。这样就实现了将网络模型的参数保存到磁盘的操作。

    # 这段代码定义了一个名为 save_networks 的函数，用于将所有网络模型保存到磁盘中。
    # 该函数接受一个参数 epoch，表示当前的训练轮数，用于在文件名中区分不同的保存版本。
    # 函数首先会遍历 self.model_names 列表中的每个网络模型名称 name，然后构造保存文件名 save_filename，其中包含了当前轮数 epoch 和网络模型名称。接着根据保存文件名和保存目录路径构造完整的保存路径 save_path。
    # 接下来，函数会通过 getattr 方法获取当前网络模型 net，这里假设网络模型的属性名按照 'net' + name 的格式命名。
    # 然后，函数会检查是否有可用的 GPU，并且至少有一个 GPU id 在 self.gpu_ids 列表中。如果满足条件，则将网络模型转移到 CPU 上并保存模型参数；然后再将模型转移到 GPU 上（假设第一个 GPU id 存在并可用）。
    # 最后，如果没有可用的 GPU，或者没有指定的 GPU id，函数将直接保存模型参数到磁盘。

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()  # OrderedDict 是 Python 标准库 collections 模块中的一个类，它是一个有序字典，会按照元素添加的顺序来保持键值对的顺序。
        for name in self.visual_names:  # 在这段代码中，visual_ret 被初始化为空的有序字典，接下来的操作可能会向其中添加键值对。
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    # 这段代码定义了一个名为 get_current_visuals 的函数，它的作用是获取当前的可视化图像并返回这些图像。
    # 具体来说，这个函数从类实例中获取 visual_names 属性，该属性是一个列表，包含了需要可视化的图像的名称。然后，对于每个名称，如果它是一个字符串类型，那么就将该名称和类实例中对应的属性值添加到字典 visual_ret 中。
    # 最终，函数返回一个有序字典 visual_ret，其中键是可视化图像的名称，值是对应的图像数据。在 train.py 脚本中，这些图像将用于显示和记录训练过程中的可视化结果。

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(
                    # 无论 getattr(self, 'loss_' + name) 返回的是什么类型的数值（可以是标量张量也可以是普通的数值），经过 float(...) 处理后都会得到相应的浮点数表示。这样可以确保最终存储到 errors_ret 字典中的值统一为浮点数，方便后续的处理和输出。
                    getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    # 这段代码看起来像是用于获取当前训练过程中的损失值或错误信息。它首先创建了一个空的有序字典 errors_ret，然后遍历 self.loss_names 中的每个元素，将损失值或错误信息添加到 errors_ret 中。最后返回这个填充好的 errors_ret 字典。
    # 这段代码中使用了 Python 的 OrderedDict 类来保持元素的插入顺序，确保了打印和保存时的顺序和遍历时的一致性。在遍历 self.loss_names 时，利用了 getattr 函数来动态获取 self 对象中以 'loss_' + name 为属性名的属性，并通过 float 函数将其转换为浮点数后存储到 errors_ret 字典中。

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']  # 通过访问参数组中的 'lr' 键，可以获取当前的学习率值。
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()  # 调用 scheduler.step() 方法会使学习率调度器按照设定的规则更新学习率，通常，该方法会根据学习率衰减策略（如指数衰减、余弦退火等）更新当前学习率值。
        new_lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate: {0} =========> {1}'.format(old_lr, new_lr))

    # 这段代码是用于更新所有网络的学习率。它会在每个 epoch 结束时被调用。
    # 代码首先获取当前学习率 old_lr，通过访问第一个优化器的参数组（self.optimizers[0].param_groups[0]）来获取学习率。接着，代码遍历所有的学习率调度器（self.schedulers），根据学习率策略进行相应的更新。
    # 如果学习率策略是 'plateau'，则调用 scheduler.step(self.metric) 来更新学习率，其中 self.metric 是作为参数传递给学习率调度器的度量指标。否则，调用 scheduler.step() 即可。
    # 最后，代码再次获取新的学习率 new_lr，并将旧学习率和新学习率打印出来，以便查看学习率的变化情况。

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """

        with torch.no_grad():
            for model in self.model_names:
                net = eval('self.net' + model)  # 这行代码的作用是根据传入的模型名称动态地获取对应的模型对象（直白一点就是说获取网络）。
                # 假设 model 是一个字符串，表示模型的名称（比如 "ResNet"、"VGG" 等），而 self.net{model} 则是一个类成员变量，表示模型对象。在这种情况下，如果想要根据 model 的值来动态获取对应的模型对象，可以使用 eval() 函数。
                # 'self.net' + model 这个表达式会将字符串 'self.net' 和模型名称 model 拼接在一起，形成一个新的字符串，比如 'self.netResNet'。然后，eval('self.net' + model) 就相当于执行了 self.netResNet，从而获得了对应的模型对象。
                net.eval()  # net.eval() 是 PyTorch 中用于将模型设置为评估模式的方法。
                # 在深度学习中，训练阶段和推断（测试）阶段有不同的模型行为，例如在推断阶段通常会关闭 Dropout 和 Batch Normalization 层的自适应行为，以便获得更稳定和一致的输出。为了实现这一点，PyTorch 提供了两种模型模式：训练模式（training mode）和评估模式（evaluation mode）。
                # 调用 net.eval() 方法可以将模型 net 设置为评估模式。在评估模式下，模型的行为会发生一些变化，比如：
                # Dropout 层会被设置为不生效，即在前向传播时不会随机丢弃神经元。
                # Batch Normalization 层会使用移动平均值和方差进行归一化，而不是根据当前 batch 的统计信息。
                # 通过将模型设置为评估模式，可以确保在推断阶段得到与训练阶段一致的结果，并且具有更好的泛化能力。
            # bin_op = BinOp(net)
            # bin_op.binarization()
            self.forward(isTrain=False)
            self.compute_visuals()
            #bin_op.restore()

    # 这段代码定义了一个名为 test 的方法，用于在测试时执行前向推断。具体来说，这个方法通过 torch.no_grad() 上下文管理器来包裹整个前向推断过程，以确保不会保存中间步骤用于反向传播，从而节省内存空间和计算资源。
    # with torch.no_grad(): 是 PyTorch 中的一个上下文管理器，用于控制是否计算梯度以及是否进行反向传播。
    # 在深度学习中，有时候我们需要在推断阶段执行前向传播，但不需要计算梯度或进行反向传播，以节省内存和计算资源。为了实现这一目的，PyTorch 提供了 torch.no_grad() 上下文管理器。
    # 当代码块被包裹在 with torch.no_grad(): 中时，PyTorch 会关闭梯度的计算和自动求导功能，即使在模型中定义了需要计算梯度的操作（比如权重更新、反向传播等）。这样可以确保在该代码块中的计算不会影响模型的梯度状态，也不会消耗额外的内存来保存梯度信息。
    #

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass
    # 在这段代码中，方法的主体部分被 pass 语句代替，意味着这个方法目前没有实际的计算逻辑，只是一个占位符。
    # 通常情况下，你需要在 compute_visuals 方法内部编写具体的计算逻辑，生成额外的输出图像数据，以供后续的可视化展示使用。这可能涉及到对模型输出的处理、图像合成、特征提取等操作，具体取决于你的应用场景和需求。


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.3)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs_decay, eta_min=0)
    elif opt.lr_policy == 'cycle-cosine':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=opt.lr_decay_iters, T_mult=1, eta_min=5.67e-6)  # T_0:第一个周期的长度（epoch 数）。   T_mult:每个后续周期的长度倍数
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler
    # 这段代码定义了一个名为 get_scheduler 的函数，用于返回一个学习率调度器（learning rate scheduler）。该函数根据传入的优化器（optimizer）和参数选项（opt），选择合适的学习率调度器并返回。
    # 如果参数选项 opt 中的 lr_policy 为 'linear'，则会定义一个 lambda_rule 函数，用于实现线性学习率衰减规则。然后创建一个 LambdaLR 学习率调度器，根据 lambda_rule 函数进行学习率调度。
    # 如果 lr_policy 为 'step'，则使用 PyTorch 默认的 StepLR 调度器，设定步长和衰减因子。
    # 如果 lr_policy 为 'plateau'，则使用 ReduceLROnPlateau 调度器，基于指定的条件来降低学习率。
    # 如果 lr_policy 为 'cosine'，则使用 CosineAnnealingLR 调度器，实现余弦退火学习率调度。
    # 如果传入的 lr_policy 不在上述四种情况内，则会返回一个 NotImplementedError，提示该学习率策略尚未实现。


# 这段代码定义了一个用于初始化神经网络权重的函数 init_net。该函数接受一些参数，如要初始化的网络 net、初始化方法 init_type、初始化增益 init_gain、GPU 设备列表 gpu_ids。
# 在函数内部，首先判断是否有指定 GPU，并将网络移动到第一个 GPU 上（如果有的话），然后使用 torch.nn.DataParallel 将网络包装成 DataParallel 类型以支持多 GPU 训练。
# 接着定义了一个局部函数 init_func(m)，用于具体的权重初始化操作。根据不同的网络层类型和指定的初始化方法，对权重进行初始化。支持的初始化方法包括 'normal'、'xavier'、'kaiming' 和 'orthogonal'。
# 最后，将初始化函数应用到网络的所有模块中，实现对整个网络的权重初始化，并返回初始化后的网络对象。
def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], ):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    # save_dir = os.path.join('./checkpoints', 'OPENet', '2025-08-05-14-09')
    # load_filename = '%s_net_%s.pth' % (3720, 'EPI')
    # load_path = os.path.join(save_dir,load_filename)
    # print('loading the pretrained weights from %s' % load_path)
    # device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')
    # state_dict = torch.load(load_path, map_location=str(device))
    # net.load_state_dict(state_dict)

    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

    return net


# 这段代码定义了一个函数 get_optimizer，用于根据不同的优化策略创建并返回相应的优化器对象。函数接受一些参数，包括 opt（可能是一些训练选项或配置）、paras（模型的可训练参数）、LR（学习率）、momentum（动量，针对某些优化器）、alpha（RMSprop 中的衰减率）。
# 函数根据传入的 opt.opti_policy 的值选择相应的优化策略，并使用给定的参数来创建对应的优化器对象。如果传入的优化策略未在函数中实现，它会返回一个 NotImplementedError。
# 这个函数的作用是根据用户指定的优化策略和参数来创建相应的优化器对象，以便在模型训练时使用。这样的设计使得可以方便地切换不同的优化策略，而无需更改大量的代码。
def get_optimizer(opt, paras, LR, momentum=0.8, alpha=0.9):
    if opt.opti_policy == 'SGD':
        my_optimizer = torch.optim.SGD(paras, lr=LR)
    elif opt.opti_policy == 'Momentum':
        my_optimizer = torch.optim.SGD(paras, lr=LR, momentum=momentum)
    elif opt.opti_policy == 'RMSprop':
        my_optimizer = torch.optim.RMSprop(paras, lr=LR, alpha=alpha)
    elif opt.opti_policy == 'Adam':
        my_optimizer = torch.optim.Adam(paras, lr=LR, betas=(opt.beta1, 0.99))
    else:
        return NotImplementedError('optimizer policy [%s] is not implemented', opt.opti_policy)
    return my_optimizer
