import torch
import torch.nn as nn
from torch.nn.functional import grid_sample,pad,softmax
import numpy as np
import torch.nn.functional as F
from math import exp

# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret

# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


IDX = [[36+j for j in range(9)],
       [4+9*j for j in range(9)],
       [10*j for j in range(9)], 
       [8*(j+1) for j in range(9)]]

index = [36,37,38,39,40,41,42,43,44,4,13,22,31,40,49,58,67,76,0,10,20,30,40,50,60,70,80,8,16,24,32,40,48,56,64,72]

class noOPAL(nn.Module):
    def __init__(self, opt,de):
        super().__init__()
        self.loss_ssim = SSIM()
        self.base_loss = torch.nn.L1Loss()
        self.views = opt.use_views
        self.patch_size = opt.input_size
        self.center_index = self.views // 2
        self.alpha = opt.alpha
        self.pad = opt.pad
        x, y = torch.arange(0, self.patch_size+2*self.pad), torch.arange(0, self.patch_size+2*self.pad)
        self.meshgrid = torch.stack(torch.meshgrid(x, y), -1).unsqueeze(0) # 1*H*W*2

    def forward(self, pred_disp,  *args): 
        total_loss = 0
        if isinstance(pred_disp, list):# raw loss
            total_loss += self.cal_l1(pred_disp)
        else: # final loss
            for i, views in enumerate(args):
                total_loss += self.cal_l1(self.warpping(pred_disp, views, IDX[i]))
        return total_loss
      

    def warpping(self, disp, views_list, idx):
        disp = disp.squeeze(1)
        B,C,H,W = views_list[0].shape
        # assert H==self.patch_size and W==self.patch_size,"size is different!"
        tmp = []
        meshgrid = self.meshgrid.repeat(B, 1, 1, 1).to(disp) # B*H*W*2
        for k in range(9):  ##############   5       7                                                                                        #############
            u, v = divmod(idx[k], 9) ######## k+2  k+1
            grid = torch.stack([ 
                torch.clip(meshgrid[:,:,:,1]-disp*(v-4),0,W-1),
                torch.clip(meshgrid[:,:,:,0]-disp*(u-4),0,H-1)
            ],-1)/(W-1) *2 -1  # B*H*W*2  归一化到-1，1
            tmp.append(grid_sample(views_list[k], grid, align_corners=True))  ######## k+2  k+1
        return tmp

    def cal_l1(self, x_list):
        
        cv_gray = x_list[4][:,0,:,:]*0.299+x_list[4][:,1,:,:]*0.587+x_list[4][:,2,:,:]*0.114
        maps = []
        loss = 0
        for j in range(9):
            loss += 0.15 * self.base_loss( x_list[j], x_list[4])
            loss_ssim = self.loss_ssim(x_list[j], x_list[4])
            loss += 0.85 * (1-loss_ssim) / 2
            if j != 4:
                maps.append(x_list[j])
        maps = torch.stack(maps,dim=1)     
        map_all = torch.mean(maps[:,:,:,:],dim=1)
        loss += 10 * self.base_loss( map_all, x_list[4])
       
        return loss
    
# class Lossv4(nn.Module):
#     def __init__(self, opt,de):
#         super().__init__()
#         self.base_loss = torch.nn.L1Loss()
#         self.views = opt.use_views
#         self.patch_size = opt.input_size
#         self.center_index = self.views // 2
#         self.alpha = opt.alpha
#         self.pad = opt.pad
#         x, y = torch.arange(0, self.patch_size+2*self.pad), torch.arange(0, self.patch_size+2*self.pad)
#         self.meshgrid = torch.stack(torch.meshgrid(x, y), -1).unsqueeze(0) # 1*H*W*2

#     def forward(self, epoch, pred_disp,  *args): 
#         '''
#         pred_disp: (tensor) B*1*H*W
#         views_x: (tensor list) [B*C*H*W]* 9
#         views_y: (tensor list) [B*C*H*W]* 9
#         views_45: (tensor list) [B*C*H*W]* 9
#         views_135: (tensor list) [B*C*H*W]* 9
#         '''
#         total_loss = 0
#         if isinstance(pred_disp, list):# raw loss
#             total_loss += self.cal_l1(pred_disp,epoch)
#         else: # final loss
#             for i, views in enumerate(args):
#                 total_loss += self.cal_l1(self.warpping(pred_disp, views, IDX[i]),epoch)
#         return total_loss
      
#     def warpping(self, disp, views_list, idx):
#         disp = disp.squeeze(1)
#         B,C,H,W = views_list[0].shape
#         # assert H==self.patch_size and W==self.patch_size,"size is different!"
#         tmp = []
#         meshgrid = self.meshgrid.repeat(B, 1, 1, 1).to(disp) # B*H*W*2
#         for k in range(9):  ##############   5       7                                                                                        #############
#             u, v = divmod(idx[k], 9) ######## k+2  k+1
#             grid = torch.stack([ 
#                 torch.clip(meshgrid[:,:,:,1]-disp*(v-4),0,W-1),
#                 torch.clip(meshgrid[:,:,:,0]-disp*(u-4),0,H-1)
#             ],-1)/(W-1) *2 -1  # B*H*W*2  归一化到-1，1
#             tmp.append(grid_sample(views_list[k], grid, align_corners=True))  ######## k+2  k+1
#         return tmp

#     def cal_l1(self, x_list, epoch):
#         maps = []
#         cv_gray = x_list[4][:,0,:,:]*0.299+x_list[4][:,1,:,:]*0.587+x_list[4][:,2,:,:]*0.114
#         rate = 1/(cv_gray.detach()+2e-1)
#         for i,x in enumerate(x_list):
#             if i != 4:
#                 map = torch.abs(x-x_list[4])
#                 map = map[:,0,:,:]*0.299+map[:,1,:,:]*0.587+map[:,2,:,:]*0.114
#                 maps.append(map)
#         maps = torch.stack(maps,dim=1)

#         map_up = torch.mean(maps[:,0:4,:,:],dim=1)
#         map_down = torch.mean(maps[:,4:,:,:],dim=1)
#         if epoch>=30: map_circle = torch.mean(maps[:,2:6,:,:],dim=1)
#         map_all = torch.mean(maps[:,:,:,:],dim=1)

#         diff = torch.abs(map_up-map_down)
#         diff = torch.where(diff<0.01,1,0)

#         if epoch>=30: maps = torch.stack([map_up,map_down,map_circle,map_all],dim=1)
#         else: maps = torch.stack([map_up,map_down,map_all],dim=1)

#         map_min,_ = torch.min(maps,dim=1)
#         loss = map_min*(1-diff)+map_all*diff
#         loss = torch.mean(loss*rate)
       
#         return loss


class Lossv4(nn.Module):
    def __init__(self, opt,de):
        super().__init__()
        self.loss_ssim = SSIM()
        self.base_loss = torch.nn.L1Loss()
        self.views = opt.use_views
        self.patch_size = opt.input_size
        self.center_index = self.views // 2
        self.alpha = opt.alpha
        self.pad = opt.pad
        x, y = torch.arange(0, self.patch_size+2*self.pad), torch.arange(0, self.patch_size+2*self.pad)
        self.meshgrid = torch.stack(torch.meshgrid(x, y), -1).unsqueeze(0) # 1*H*W*2

    def forward(self, epoch, pred_disp, *args): 
        '''
        pred_disp: (tensor) B*1*H*W
        views_x: (tensor list) [B*C*H*W]* 9
        views_y: (tensor list) [B*C*H*W]* 9
        views_45: (tensor list) [B*C*H*W]* 9
        views_135: (tensor list) [B*C*H*W]* 9
        '''
        total_loss = 0
        if isinstance(pred_disp, list):# raw loss
            total_loss += self.cal_l1(pred_disp,epoch)
        else: # final loss
            for i, views in enumerate(args):
                total_loss += self.cal_l1(self.warpping(pred_disp, views, IDX[i]),epoch)
        return total_loss
    
    # def warpping(self, disp, views_list, idx):
    #     B,C,H,W = views_list[0].shape
    #     disp = disp.squeeze(1)
    #     tmp = []
    #     meshgrid = self.meshgrid.repeat(B, 1, 1, 1).to(disp) # B*H*W*2
    #     for k in range(81):
    #         u, v = divmod(k, 9)
    #         grid = torch.stack([ 
    #             torch.clip(meshgrid[:,:,:,1]-disp*(v-4),0,W-1),
    #             torch.clip(meshgrid[:,:,:,0]-disp*(u-4),0,H-1)
    #         ],-1)/(W-1) *2 -1  # B*H*W*2  归一化到-1，1
    #         tmp.append(grid_sample(views_list[k], grid, align_corners=True))   
    #     return tmp
    
    def warpping(self, disp, views_list, idx):
        disp = disp.squeeze(1)
        B,C,H,W = views_list[0].shape
        # assert H==self.patch_size and W==self.patch_size,"size is different!"
        tmp = []
        meshgrid = self.meshgrid.repeat(B, 1, 1, 1).to(disp) # B*H*W*2
        for k in range(9):  ##############   5       7                                                                                        #############
            u, v = divmod(idx[k], 9) ######## k+2  k+1
            grid = torch.stack([ 
                torch.clip(meshgrid[:,:,:,1]-disp*(v-4),0,W-1),
                torch.clip(meshgrid[:,:,:,0]-disp*(u-4),0,H-1)
            ],-1)/(W-1) *2 -1  # B*H*W*2  归一化到-1，1
            tmp.append(grid_sample(views_list[k], grid, align_corners=True))  ######## k+2  k+1
        return tmp

    
    def cal_l1(self, x_list, epoch):
        maps = []
        masks = []
        cv_gray = x_list[4][:,0,:,:]*0.299+x_list[4][:,1,:,:]*0.587+x_list[4][:,2,:,:]*0.114  # [2, 64, 64]    
        # cv_gray = x_list[4]  # [2, 3, 64, 64]
        rate = 1/(cv_gray.detach()+2e-1)  # 2e-1表示2*0.1=0.2，类似于7e-4 ，rate用来放大损失，不然损失太小了
        for i,x in enumerate(x_list):
            # if i != 40:
            map = torch.abs(x-x_list[4])  # [2, 3, 64, 64]
            map = map[:,0,:,:]*0.299+map[:,1,:,:]*0.587+map[:,2,:,:]*0.114  # [2, 64, 64]
            # t = torch.mean(map)
            q = torch.tensor([0.75]).to(map)
            t = torch.quantile(map,q=q)
            # 不能这样直接比较，因为warp之后可能在相邻像素点，虽然损失小，但其实是错误的视差,但是OAVC可以根据候选视差进行投票，而这边只有这一个生成的视差
            mask = torch.where(map<t,1,0)  # 量化方法的坏像素太大了，0.07的坏像素都很大，因此该阈值不能太小。坏像素太大的那些噪点在蒸馏里面的损失是不是应该排除
            map = map*mask
            masks.append(mask)
            maps.append(map)
        # 在所有非0区域求平均
        # masks = torch.stack(masks,dim=1) 
        # mask_sum = torch.sum(masks, dim=1)
        # epsilon = 1  # 一定要避免除以零或者除以很小的值，例如1e-8，不然会生成NaN值，因此在 mask_sum 为 0 的位置上用 1 代替
        # safe_mask_sum = torch.where(mask_sum > 0, mask_sum, torch.full_like(mask_sum, epsilon))
        # maps = torch.stack(maps,dim=1)  # [2, 8, 3, 64, 64]  [2, 8, 64, 64]
        # weighted_sum = torch.sum(maps, dim=1) 
        # loss = torch.where(mask_sum > 0, weighted_sum / safe_mask_sum, torch.zeros_like(weighted_sum))  
        # loss = torch.mean(loss*rate)
        # 直接求平均
        maps = torch.stack(maps,dim=1)
        loss = torch.mean(maps, dim=1)
        loss = torch.mean(loss*rate)
        
        
#         # noOPAL
#         loss = 0
#         for j in range(49):
#             loss += self.base_loss( x_list[j], x_list[4])
        
#         # OccCasNet遮挡（这个方法的软掩码还挺相似的和我的方法）
#         # loss = 0
#         # for j in range(49):
#         #     mask = torch.clip(torch.abs(x_list[j] - x_list[24]),0,1)
#         #     loss += self.base_loss( x_list[j]*(1-mask), x_list[24]*(1-mask))
        
#         return loss
        
        if epoch<=10: 
            maps_noOPAL = []
            loss = 0
            for j in range(9):
                loss += 0.15 * self.base_loss( x_list[j], x_list[4])
                loss_ssim = self.loss_ssim(x_list[j], x_list[4])
                loss += 0.85 * (1-loss_ssim) / 2
                if j != 4:
                    maps_noOPAL.append(x_list[j])
            maps_noOPAL = torch.stack(maps_noOPAL,dim=1)     
            map_all = torch.mean(maps_noOPAL[:,:,:,:],dim=1)
            loss += 10 * self.base_loss( map_all, x_list[4])


        
        # np.save("map_up.npy", map_up[0,:,:].detach().cpu().numpy())
        # np.save("map_down.npy", map_down[0,:,:].detach().cpu().numpy())
        # np.save("map_circle.npy", map_circle[0,:,:].detach().cpu().numpy())
        # np.save("map_all.npy", map_all[0,:,:].detach().cpu().numpy())
        
        # diff = torch.abs(map_up-map_down)
        # diff = torch.where(diff<0.01,1,0)
        # mask = torch.where(map_up<0.01,1,0)
        # mask = diff*(1-mask)  # 两侧都存在遮挡的mask
        # 至少一半以上的区域被认为是没有遮挡的，可事实上，在这些区域中，有4%~35%的区域存在对称遮挡
        # num_ones = torch.sum(mask == 1).item()
        # total_elements = mask.numel()
        # percentage_ones = (num_ones / total_elements) * 100
        # print(f"Percentage of 1s: {percentage_ones:.2f}%")
        
        # np.save("mask.npy", mask[0,:,:].detach().cpu().numpy())

        # map_min,index = torch.min(maps,dim=1)
        # index = (index+1)*(1-diff)
        # # 使用 bincount 统计每个索引出现的次数
        # # 由于 bincount 需要一维输入，我们先要展平 index 张量
        # counts = torch.bincount(index.view(-1))
        # # 打印每个索引的出现次数
        # for i, count in enumerate(counts):
        #     print(f"Index {i} appears {count} times")
       
        return loss

    
# 用教师模型生成mask
# class Lossv4(nn.Module):
#     def __init__(self, opt,de):
#         super().__init__()
#         self.loss_ssim = SSIM()
#         self.base_loss = torch.nn.L1Loss()
#         self.views = opt.use_views
#         self.patch_size = opt.input_size
#         self.center_index = self.views // 2
#         self.alpha = opt.alpha
#         self.pad = opt.pad
#         x, y = torch.arange(0, self.patch_size+2*self.pad), torch.arange(0, self.patch_size+2*self.pad)
#         self.meshgrid = torch.stack(torch.meshgrid(x, y), -1).unsqueeze(0) # 1*H*W*2

#     def forward(self, epoch, pred_disp, input0, input1, input2, input3, label): 
#         total_loss = 0
#         if isinstance(pred_disp, list):# raw loss
#             total_loss += self.cal_l1(pred_disp[0],epoch,self.warpping(label, input0, IDX[0]))
#             total_loss += self.cal_l1(pred_disp[1],epoch,self.warpping(label, input1, IDX[1]))
#             total_loss += self.cal_l1(pred_disp[2],epoch,self.warpping(label, input2, IDX[2]))
#             total_loss += self.cal_l1(pred_disp[3],epoch,self.warpping(label, input3, IDX[3]))
#         else:
#             total_loss += self.cal_l1(self.warpping(pred_disp, input0, IDX[0]),epoch,self.warpping(label, input0, IDX[0]))
#             total_loss += self.cal_l1(self.warpping(pred_disp, input1, IDX[1]),epoch,self.warpping(label, input1, IDX[1]))
#             # total_loss += self.cal_l1(self.warpping(pred_disp, input2, IDX[2]),epoch,self.warpping(label, input2, IDX[2]))
#             # total_loss += self.cal_l1(self.warpping(pred_disp, input3, IDX[3]),epoch,self.warpping(label, input3, IDX[3]))
#         return total_loss

#     def warpping(self, disp, views_list, idx):
#         disp = disp.squeeze(1)
#         B,C,H,W = views_list[0].shape
#         # assert H==self.patch_size and W==self.patch_size,"size is different!"
#         tmp = []
#         meshgrid = self.meshgrid.repeat(B, 1, 1, 1).to(disp) # B*H*W*2
#         for k in range(9):  ##############   5       7                                                                                        #############
#             u, v = divmod(idx[k], 9) ######## k+2  k+1
#             grid = torch.stack([ 
#                 torch.clip(meshgrid[:,:,:,1]-disp*(v-4),0,W-1),
#                 torch.clip(meshgrid[:,:,:,0]-disp*(u-4),0,H-1)
#             ],-1)/(W-1) *2 -1  # B*H*W*2
#             tmp.append(grid_sample(views_list[k], grid, align_corners=True))  ######## k+2  k+1
#         return tmp
    
#     def cal_l1(self, x_list, epoch,mask_list):
#         maps = []
#         masks = []
#         cv_gray = x_list[4][:,0,:,:]*0.299+x_list[4][:,1,:,:]*0.587+x_list[4][:,2,:,:]*0.114  # [2, 64, 64]
#         rate = 1/(cv_gray.detach()+2e-1)
#         for i,x in enumerate(mask_list):
#             map = torch.abs(x-mask_list[4])  # [2, 3, 64, 64]
#             map = map[:,0,:,:]*0.299+map[:,1,:,:]*0.587+map[:,2,:,:]*0.114  # [2, 64, 64]
#             q = torch.tensor([0.75]).to(map)
#             t = torch.quantile(map,q=q)
#             # t = torch.mean(map)
#             mask = torch.where(map<t,1,0)
#             masks.append(mask)
#         for i,x in enumerate(x_list):
#             map = torch.abs(x-x_list[4])  # [2, 3, 64, 64]
#             map = map[:,0,:,:]*0.299+map[:,1,:,:]*0.587+map[:,2,:,:]*0.114  # [2, 64, 64]
#             map = map*masks[i]
#             maps.append(map)
#         maps = torch.stack(maps,dim=1)
#         loss = torch.mean(maps, dim=1)
#         loss = torch.mean(loss*rate)
       
#         return loss


# 教师模型使用的损失，包含了SSIM
# class Lossv4(nn.Module):
#     def __init__(self, opt,de):
#         super().__init__()
#         self.loss_ssim = SSIM()
#         self.base_loss = torch.nn.L1Loss()
#         self.views = opt.use_views
#         self.patch_size = opt.input_size
#         self.center_index = self.views // 2
#         self.alpha = opt.alpha
#         self.pad = opt.pad
#         x, y = torch.arange(0, self.patch_size+2*self.pad), torch.arange(0, self.patch_size+2*self.pad)
#         self.meshgrid = torch.stack(torch.meshgrid(x, y), -1).unsqueeze(0) # 1*H*W*2

#     def forward(self, epoch, pred_disp, input0, input1, input2, input3, label): 
#         total_loss = 0
#         if isinstance(pred_disp, list):
#             total_loss += self.cal_l1(pred_disp[0],epoch,self.warpping(label, input0, IDX[0]))
#             total_loss += self.cal_l1(pred_disp[1],epoch,self.warpping(label, input1, IDX[1]))
#             total_loss += self.cal_l1(pred_disp[2],epoch,self.warpping(label, input2, IDX[2]))
#             total_loss += self.cal_l1(pred_disp[3],epoch,self.warpping(label, input3, IDX[3]))
#         else:
#             total_loss += self.cal_l1(self.warpping(pred_disp, input0, IDX[0]),epoch,self.warpping(label, input0, IDX[0]))
#             total_loss += self.cal_l1(self.warpping(pred_disp, input1, IDX[1]),epoch,self.warpping(label, input1, IDX[1]))
#             total_loss += self.cal_l1(self.warpping(pred_disp, input2, IDX[2]),epoch,self.warpping(label, input2, IDX[2]))
#             total_loss += self.cal_l1(self.warpping(pred_disp, input3, IDX[3]),epoch,self.warpping(label, input3, IDX[3]))
#         return total_loss
    
#     def warpping(self, disp, views_list, idx):
#         disp = disp.squeeze(1)
#         B,C,H,W = views_list[0].shape
#         # assert H==self.patch_size and W==self.patch_size,"size is different!"
#         tmp = []
#         meshgrid = self.meshgrid.repeat(B, 1, 1, 1).to(disp) # B*H*W*2
#         for k in range(9):  ##############   5       7                                                                                        #############
#             u, v = divmod(idx[k], 9) ######## k+2  k+1
#             grid = torch.stack([ 
#                 torch.clip(meshgrid[:,:,:,1]-disp*(v-4),0,W-1),
#                 torch.clip(meshgrid[:,:,:,0]-disp*(u-4),0,H-1)
#             ],-1)/(W-1) *2 -1  # B*H*W*2
#             tmp.append(grid_sample(views_list[k], grid, align_corners=True))  ######## k+2  k+1
#         return tmp
    
#     def cal_l1(self, x_list, epoch,mask_list):
#         maps = []
#         masks = []
#         cv_gray = x_list[4][:,0,:,:]*0.299+x_list[4][:,1,:,:]*0.587+x_list[4][:,2,:,:]*0.114  # [2, 64, 64]
#         rate = 1/(cv_gray.detach()+2e-1)
#         for i,x in enumerate(mask_list):
#             map = torch.abs(x-mask_list[4])  # [2, 3, 64, 64]
#             map = map[:,0,:,:]*0.299+map[:,1,:,:]*0.587+map[:,2,:,:]*0.114  # [2, 64, 64]
#             q = torch.tensor([0.75]).to(map)
#             t = torch.quantile(map,q=q)
#             mask = torch.where(map<t,1,0)
#             mask = mask.unsqueeze(1)
#             mask = mask.repeat(1, 3, 1, 1)
#             masks.append(mask)
#         loss = 0
#         for j in range(9):
#             loss += 0.45 * self.base_loss(x_list[j]*masks[j], x_list[4]*masks[j])
#             loss_ssim = self.loss_ssim(x_list[j]*masks[j], x_list[4]*masks[j])
#             loss += 0.7 * (1-loss_ssim) / 2
       
#         return loss




def get_smooth_loss(disp, img, lamda):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-lamda*grad_img_x)
    grad_disp_y *= torch.exp(-lamda*grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


# 定义蒸馏损失函数
def get_distillation_loss(student_outputs, teacher_outputs, epoch, alpha):

    # KL散度损失
#     temperature = 8
#     alpha_distillation_loss = 5  # 蒸馏损失的权重，用于平衡蒸馏损失和原始损失之间的关系。
    
#     batch_size = student_outputs.size(0)
#     channel = student_outputs.size(1)
#     student_softmax = nn.functional.softmax(student_outputs.reshape(batch_size,channel,-1) / temperature, dim=2)  #  将输出形状为torch.Size([2, 1, 64, 64])的张量重塑之后再进行softmax
#     teacher_softmax = nn.functional.softmax(teacher_outputs.reshape(batch_size,channel,-1) / temperature, dim=2)
#     kl_div = nn.functional.kl_div(student_softmax.log(), teacher_softmax, reduction='batchmean')  # 计算学生模型的 Softmax 输出和教师模型的 Softmax 输出之间的 KL 散度，并对整个批次的 KL 散度求平均。
    
#     return kl_div * (temperature**2) * alpha_distillation_loss * alpha

    # L1损失
    criterion  = torch.nn.L1Loss()
    L1_loss = criterion(student_outputs,teacher_outputs)
    
    return L1_loss * alpha
