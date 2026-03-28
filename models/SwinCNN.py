from models.ours_encoder import SwinTransformer,SwinCNN,SwinCNNFusion
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class OursSwinCNN(nn.Module):
    def __init__(self,dec_channel = 256, **kwargs):
        super(OursSwinCNN, self).__init__()

        self.encoder = nn.Sequential(
            SpectralCalibration(kwargs['n_bands'], 256),
            SwinCNN(img_size=kwargs['patch_size'], in_chans=256)
        )
        self.shared_fea_dim = 96

        # 分类
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

    def forward(self,x):
        shared_fea_cls = self.encoder(x)
        feature_cls = self.cnn_cls(shared_fea_cls)
        feature_cls_mid1 = self.trans1_cls.forward2(feature_cls)
        feature_cls_mid2 = self.trans2_cls.forward2(feature_cls_mid1)
        cls = self.head_cls(feature_cls_mid2)
        return cls



if __name__ == "__main__":
    hyper = {
        'patch_size': 16,
        'n_bands': 200,
        'n_classes': 17
    }
    x = torch.rand(10, 200, 16, 16)
    # x_lr = torch.rand(10, 200, 8, 8)
    # x_bic = torch.rand(10, 200, 16, 16)
    net = SwinCNN(**hyper)
    y = net(x)
    print(y.shape) # ,y[2].shape,y[3].shape
    #
    # import numpy as np
    # ignored_mask = np.zeros((2,2), dtype=bool)
    # print(ignored_mask)