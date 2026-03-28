import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torchvision.models.vgg import vgg16
from torchvision.models.vgg import vgg19


class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()

    def forward(self, x):
        x0 = x[:, 0]

        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)

        return x0


class spatial_grad(nn.Module):
    def __init__(self, weight):
        super(spatial_grad, self).__init__()
        self.get_grad = Get_gradient_nopadding()
        self.fidelity = torch.nn.L1Loss()
        self.weight = weight

    def forward(self, y, gt):
        y_grad = self.get_grad(y)
        gt_grad = self.get_grad(gt)
        return self.weight * self.fidelity(y_grad, gt_grad)


class SAMLoss(torch.nn.Module):
    def __init__(self):
        super(SAMLoss, self).__init__()

    def forward(self, y, gt):
        B, C, H, W = y.shape
        y_flat = y.reshape(B, C, -1)
        gt_flat = gt.reshape(B, C, -1)
        y_norm = torch.norm(y_flat, 2, dim=1)
        gt_norm = torch.norm(gt_flat, 2, dim=1)
        numerator = torch.sum(gt_flat*y_flat, dim=1)
        denominator = y_norm * gt_norm
        sam = torch.div(numerator, denominator + 1e-5)
        sam = torch.sum(torch.acos(sam)) / (B * H * W) * 180 / np.pi  # 3.14159
        return sam

class HybridLoss(torch.nn.Module):
    def __init__(self, lamd=1e-1, spatial_tv=False, spectral_tv=False):
        super(HybridLoss, self).__init__()
        self.lamd = lamd
        self.use_spatial_TV = spatial_tv
        self.use_spectral_TV = spectral_tv
        self.fidelity = torch.nn.L1Loss()
        self.spatial = TVLoss(weight=1e-3)
        self.spectral = TVLossSpectral(weight=1e-3)

    def forward(self, y, gt):
        loss = self.fidelity(y, gt)
        spatial_TV = 0.0
        spectral_TV = 0.0
        if self.use_spatial_TV:
            spatial_TV = self.spatial(y)
        if self.use_spectral_TV:
            spectral_TV = self.spectral(y)
        total_loss = loss + spatial_TV + spectral_TV
        return total_loss


# from https://github.com/jxgu1016/Total_Variation_Loss.pytorch with slight modifications
class TVLoss(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        # h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :h_x - 1, :]).sum()
        # w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x - 1]).sum()
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class TVLossSpectral(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(TVLossSpectral, self).__init__()
        self.TVLoss_weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        c_x = x.size()[1]
        count_c = self._tensor_size(x[:, 1:, :, :])
        # c_tv = torch.abs((x[:, 1:, :, :] - x[:, :c_x - 1, :, :])).sum()
        c_tv = torch.pow((x[:, 1:, :, :] - x[:, :c_x - 1, :, :]), 2).sum()
        return self.TVLoss_weight * 2 * (c_tv / count_c) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def cal_gradient_c(x):
    c_x = x.size(1)
    g = x[:, 1:, 1:, 1:] - x[:, :c_x - 1, 1:, 1:]
    return g


def cal_gradient_x(x):
    c_x = x.size(2)
    g = x[:, 1:, 1:, 1:] - x[:, 1:, :c_x - 1, 1:]
    return g


def cal_gradient_y(x):
    c_x = x.size(3)
    g = x[:, 1:, 1:, 1:] - x[:, 1:, 1:, :c_x - 1]
    return g


def cal_gradient(inp):
    x = cal_gradient_x(inp)
    y = cal_gradient_y(inp)
    c = cal_gradient_c(inp)
    g = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2) + torch.pow(c, 2) + 1e-6)
    return g


def cal_sam(Itrue, Ifake):
    esp = 1e-6
    # element-wise product
    # torch.sum(dim=1) 沿通道求和
    # [B C H W] * [B C H W] --> [B C H W]  Itrue*Ifake
    # [B 1 H W] InnerPro(keepdim)

    InnerPro = torch.sum(Itrue * Ifake, 1, keepdim=True)
    # print('InnerPro')
    # print(InnerPro.shape)
    # 沿通道求范数
    # len1  len2  [B 1 H W] (keepdim)
    len1 = torch.norm(Itrue, p=2, dim=1, keepdim=True)
    len2 = torch.norm(Ifake, p=2, dim=1, keepdim=True)
    # print('len1')
    # print(len1.shape)

    divisor = len1 * len2
    mask = torch.eq(divisor, 0)
    divisor = divisor + (mask.float()) * esp
    cosA = torch.sum(InnerPro / divisor, 1).clamp(-1 + esp, 1 - esp)
    # print(cosA.shape)
    sam = torch.acos(cosA)
    sam = torch.mean(sam) / np.pi
    return sam


# <A Group-Based Embedding Learning and Integration Network for Hyperspectral Image Super-Resolution> 的loss
class HLoss(torch.nn.Module):
    def __init__(self, sam_weight = 0.5, gra_weight = 0.1): # 0.5,0.1
        super(HLoss, self).__init__()
        self.lamd1 = sam_weight
        self.lamd2 = gra_weight

        self.fidelity = torch.nn.L1Loss()
        self.gra = torch.nn.L1Loss()

        # self.spatial = TVLoss(weight=tv_weight)
        # self.spectral = TVLossSpectral(weight=tv_weight)

    def forward(self, y, gt):
        loss1 = self.fidelity(y, gt)
        loss2 = cal_sam(y, gt)
        loss3 = self.gra(cal_gradient(y), cal_gradient(gt))
        # spatial_TV = self.spatial(y)
        # spectral_TV = self.spectral(y)

        loss = loss1 + self.lamd1 *loss2 + self.lamd2 * loss3 #+ spatial_TV + spectral_TV
        return loss

class HLoss_MSDformer(torch.nn.Module):
    def __init__(self, sam_weight = 0.5, gra_weight = 0.1, tv_weight = 1e-3): # 0.5,0.1
        super(HLoss_MSDformer, self).__init__()
        self.lamd1 = sam_weight
        self.lamd2 = gra_weight

        self.fidelity = torch.nn.L1Loss()
        self.gra = torch.nn.L1Loss()

    def forward(self, y, gt):
        loss1 = self.fidelity(y, gt)
        loss2 = cal_sam(y, gt)
        loss3 = self.gra(cal_gradient(y), cal_gradient(gt))

        loss = loss1 + self.lamd1 *loss2 + self.lamd2 * loss3
        return loss
# SRGAN的loss
class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        # vgg = vgg16(pretrained=True)
        # loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        # for param in loss_network.parameters():
        #     param.requires_grad = False
        # self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.l1loss = nn.L1Loss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss
        # perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))

        # SAM LOSS == Spectral Loss 将SAMloss替换Perceptionloss
        sam_loss = SAMLoss()(out_images,target_images)

        # Image Loss
        image_loss = self.l1loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        # return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss  # 原SRGANloss
        return image_loss + 0.001 * adversarial_loss + 0.01 * sam_loss + 1e-6 * tv_loss

# ESRGAN
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        vgg = vgg19(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:35]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.l1_loss = nn.L1Loss()

    def forward(self, high_resolution, fake_high_resolution):
        perception_loss = self.l1_loss(self.loss_network(high_resolution), self.loss_network(fake_high_resolution))
        return perception_loss


# GDRRN loss
class myloss_spe(nn.Module):
    def __init__(self, N, lamd = 1e-1, mse_lamd=1, epoch=None):
        super(myloss_spe, self).__init__()
        self.N = N
        self.lamd = lamd
        self.mse_lamd = mse_lamd
        self.epoch = epoch
        return

    def forward(self, res, label):
        mse = F.mse_loss(res, label, size_average=False)
        # mse = func.l1_loss(res, label, size_average=False)
        loss = mse / (self.N * 2)
        esp = 1e-12
        H = label.size()[2]
        W = label.size()[3]
        Itrue = label.clone()
        Ifake = res.clone()
        nom = torch.mul(Itrue, Ifake).sum(dim=1)
        denominator = Itrue.norm(p=2, dim=1, keepdim=True).clamp(min=esp) * \
                      Ifake.norm(p=2, dim=1, keepdim=True).clamp(min=esp)
        denominator = denominator.squeeze()
        # sam = -np.pi/2*torch.div(nom, denominator) + np.pi/2
        sam = torch.div(nom, denominator).acos()
        sam[sam!=sam] = 0
        sam_sum = torch.sum(sam) / (self.N * H * W)
        if self.epoch is None:
            total_loss = self.mse_lamd * loss + self.lamd * sam_sum
        else:
            norm = self.mse_lamd + self.lamd * 0.1 **(self.epoch//10)
            lamd_sam = self.lamd * 0.1 ** (self.epoch // 10)
            total_loss = self.mse_lamd/norm * loss + lamd_sam/norm * sam_sum
        return total_loss