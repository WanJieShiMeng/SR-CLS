from models.ours_encoder import SwinTransformer,SwinCNN,SwinCNNFusion
import torch
import torch.nn as nn
from contrastive_loss import FFTSimCLR
import math
from einops import rearrange
import torch.nn.functional as F

class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class SpectralCalibration(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class AttentionSpatial(nn.Module):
    def __init__(self, dim=64, num_heads=8, bias=False):
        super(AttentionSpatial, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.q = nn.Conv2d(dim, dim , kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        kv = self.kv(y)
        k, v = kv.chunk(2, dim=1)
        q = self.q(x)

        q = rearrange(q, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head (h w) c', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head (h w) c -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)

        return out


class AttentionChannel(nn.Module):
    def __init__(self, dim=64, num_heads=8, bias=False):
        super(AttentionChannel, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim , kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(x))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class TaskCross(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.amp_attn = AttentionChannel(dim)
        self.pha_attn = AttentionChannel(dim)

        self.post = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, fea_cls, fea_sr):
        if fea_cls.shape[-1] != fea_sr.shape[-1]:
            fea_cls = F.interpolate(fea_cls, size=[i for i in fea_sr.size()[2:]], mode='bilinear', align_corners=True)
        _,_,H,W = fea_cls.shape

        fre_cls = torch.fft.rfft2(fea_cls+1e-8, norm='backward')
        amp_cls = torch.abs(fre_cls)
        pha_cls = torch.angle(fre_cls)

        fre_sr = torch.fft.rfft2(fea_sr+1e-8, norm='backward')
        amp_sr = torch.abs(fre_sr)
        pha_sr = torch.angle(fre_sr)

        amp_fuse = self.amp_attn(amp_cls,amp_sr)
        pha_fuse = self.pha_attn(pha_cls,pha_sr)

        real = amp_fuse * torch.cos(pha_fuse) + 1e-8
        imag = amp_fuse * torch.sin(pha_fuse) + 1e-8
        fea_fuse = torch.complex(real, imag) + 1e-8
        fea_fuse = torch.abs(torch.fft.irfft2(fea_fuse, s=(H, W), norm='backward'))
        fea_fuse = self.post(fea_fuse)

        out = fea_cls + fea_fuse

        return out


class MultiTask(nn.Module):
    def __init__(self,dec_channel = 256, Swin = False, fusion = True, cross = True, contrast=True, **kwargs):
        super(MultiTask, self).__init__()
        self.Swin = Swin
        self.fusion = fusion
        self.cross = cross
        self.contrast = contrast

        self.shared_fea_dim = None

        if self.Swin:
            self.encoder = nn.Sequential(
                SpectralCalibration(kwargs['n_bands'], 256),
                SwinTransformer(img_size=kwargs['patch_size'], in_chans=256)
            )
            self.shared_fea_dim = 288
        elif self.fusion:
                self.encoder = nn.Sequential(
                    SpectralCalibration(kwargs['n_bands'], 256),
                    SwinCNNFusion(img_size=kwargs['patch_size'], in_chans=256)
                    # FreSplitEncoder(img_size=kwargs['patch_size'], in_chans=256)
                )
                self.shared_fea_dim = 768#768
        else:
            self.encoder = nn.Sequential(
                SpectralCalibration(kwargs['n_bands'], 256),
                SwinCNN(img_size=kwargs['patch_size'], in_chans=256)
            )
            self.shared_fea_dim = 96#768
        # cls
        self.cnn_cls = nn.Conv2d(self.shared_fea_dim, dec_channel, 3, 1, 1) # 1344 96
        self.trans1_cls = SwinTransformer(img_size=8, patch_size=1, in_chans=dec_channel, embed_dim=dec_channel,
                             depths=(2,),
                             num_heads=(4,), window_size=2, drop_rate=0.0, out_indices=(0,))
        self.trans2_cls = SwinTransformer(img_size=8, patch_size=1, in_chans=dec_channel, embed_dim=dec_channel,
                                           depths=(2,),
                                           num_heads=(4,), window_size=2, drop_rate=0.0, out_indices=(0,))
        self.head_cls = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(dec_channel, kwargs["n_classes"])
        )

        # sr
        self.cnn_sr = nn.Conv2d(self.shared_fea_dim, dec_channel, 3, 1, 1)
        self.trans1_sr = SwinTransformer(img_size=8, patch_size=1, in_chans=dec_channel, embed_dim=dec_channel,
                                           depths=(2,),
                                           num_heads=(4,), window_size=2, drop_rate=0.0, out_indices=(0,))
        self.trans2_sr = SwinTransformer(img_size=8, patch_size=1, in_chans=dec_channel, embed_dim=dec_channel,
                                           depths=(2,),
                                           num_heads=(4,), window_size=2, drop_rate=0.0, out_indices=(0,))
        self.head_sr = nn.Sequential(
            # Upsample(2, dec_channel),
            nn.Conv2d(dec_channel, kwargs["n_bands"], 3, 1, 1)
        )
        self.skip_conv = nn.Conv2d(kwargs['n_bands'],dec_channel,3,1,1)

        # cross-task
        if self.cross:
            self.cross_net = TaskCross(dec_channel)

        # cl loss
        if self.contrast:
            self.SimCLR = FFTSimCLR(in_chan=self.shared_fea_dim) # dec_channel
            # self.SimSiam = FFTSimSiam(in_chan=self.shared_fea_dim)

    def forward(self,x):
        srFactor = 2
        loss = None

        shared_fea_cls = self.encoder(x)
        lr_x = F.interpolate(x,size=[i//srFactor for i in x.size()[2:]], mode='bilinear', align_corners=True)
        bic_x = F.interpolate(lr_x,size=[i*srFactor for i in lr_x.size()[2:]], mode='bilinear', align_corners=True)
        shared_fea_sr = self.encoder(bic_x)

        feature_cls = self.cnn_cls(shared_fea_cls)
        feature_cls_mid1 = self.trans1_cls.forward2(feature_cls)

        feature_sr = self.cnn_sr(shared_fea_sr)
        feature_sr_mid1 = self.trans1_sr.forward2(feature_sr)

        if self.cross:
            feature_cls_mid1_new = self.cross_net(feature_cls_mid1, feature_sr_mid1)
            feature_cls_mid2 = self.trans2_cls.forward2(feature_cls_mid1_new)
        else:
            feature_cls_mid2 = self.trans2_cls.forward2(feature_cls_mid1)

        feature_sr_mid2 = self.trans2_sr.forward2(feature_sr_mid1)

        cls = self.head_cls(feature_cls_mid2)
        # sr = self.head_sr(feature_sr_mid2)
        sr = self.head_sr(feature_sr_mid2 + self.skip_conv(bic_x))

        if self.contrast:
            loss = self.SimCLR(shared_fea_cls, shared_fea_sr)
            # loss = self.SimSiam(shared_fea_cls, shared_fea_sr)
        return cls, sr, loss

    # 测试的时候采用原图像输入
    def forward_cls(self, x):
        shared_fea_cls = self.encoder(x)

        bic_x = F.interpolate(x, size=[i * 2 for i in x.size()[2:]], mode='bilinear', align_corners=True)
        shared_fea_sr = self.encoder(bic_x)

        feature_cls = self.cnn_cls(shared_fea_cls)
        feature_cls_mid1 = self.trans1_cls.forward2(feature_cls)

        feature_sr = self.cnn_sr(shared_fea_sr)
        feature_sr_mid1 = self.trans1_sr.forward2(feature_sr)

        if self.cross:
            feature_cls_mid1_new = self.cross_net(feature_cls_mid1, feature_sr_mid1)
            feature_cls_mid2 = self.trans2_cls.forward2(feature_cls_mid1_new)
        else:
            feature_cls_mid2 = self.trans2_cls.forward2(feature_cls_mid1)

        cls = self.head_cls(feature_cls_mid2)

        return cls

    def forward_sr(self, x):
        shared_fea = self.encoder(x)
        feature_sr = self.cnn_sr(shared_fea)
        feature_sr_mid1 = self.trans1_sr.forward2(feature_sr)
        feature_sr_mid2 = self.trans2_sr.forward2(feature_sr_mid1)
        sr = self.head_sr(feature_sr_mid2 + self.skip_conv(x))
        # sr = self.head_sr(feature_sr_mid2)
        return sr



class MultiTask_loss(nn.Module):
    def __init__(self, cls_weight, cls_ratio=1, sr_ratio=0.1):
        super().__init__()
        # self.adaptive_loss = AutomaticWeightedLoss(2)

        self.alpha = cls_ratio
        self.beta = sr_ratio

        self.sr_loss = nn.L1Loss()
        self.cls_loss = nn.CrossEntropyLoss(weight=cls_weight)

    def forward(self,cls_out,sr_out,cls_gt,sr_gt): # ,fea_cls,fea_sr
        loss_cls = self.cls_loss(cls_out,cls_gt)
        loss_sr = self.sr_loss(sr_out,sr_gt)

        # loss_cl = 0 #self.lamd_cl * self.cl_loss(fea_cls,fea_sr)

        # total_loss = loss_cls + loss_sr #+ loss_cl

        # total_loss = self.adaptive_loss(loss_cls,loss_sr)
        total_loss = self.alpha * loss_cls + self.beta * loss_sr
        print('cls:{:.3f},sr:{:.3f}'.format(loss_cls.item(), loss_sr.item()))
        return total_loss


if __name__ == "__main__":
    hyper = {
        'patch_size': 16,
        'n_bands': 200,
        'n_classes': 17
    }
    x = torch.rand(5, 200, 16, 16)



    # x_lr = torch.rand(10, 200, 8, 8)
    # x_bic = torch.rand(10, 200, 16, 16)
    net = MultiTask(Swin = False, fusion = True, cross = True, contrast=True,**hyper)
    # print(net.encoder)
    y = net(x)
    # from thop import profile
    # Flops, params = profile(net, inputs=(x,))  # macs
    # print('Flops: % .4fG' % (Flops / 1000000000))  # 计算量
    # print('params参数量: % .4fM' % (params / 1000000))  # 参数量：等价与上面的summary输出的Total params值


    print(y[0].shape,y[1].shape,bool(y[2])) # ,y[2].shape,y[3].shape
    #
    # print(net.forward_cls2(x).shape)
    # print(net.forward_sr(x).shape)
    #
    # import numpy as np
    # ignored_mask = np.zeros((2,2), dtype=bool)
    # print(ignored_mask) 12