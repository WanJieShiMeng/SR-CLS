import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


class CNN_branch(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.Conv3d(channels, channels,
                                           kernel_size=3, padding=1,
                                           groups=channels),
                                 nn.Conv3d(channels, channels, kernel_size=1),
                                 nn.BatchNorm3d(channels),
                                 nn.ReLU6(),
                                 )

    def forward(self, x):
        return x + self.net(x)


class Attention(nn.Module):
    def __init__(self, heads, patch, drop):
        super().__init__()
        self.heads = heads
        self.patch = patch
        self.scale = patch ** -1
        self.conv_project = nn.Sequential(nn.Conv3d(1, 3 * heads,
                                                    kernel_size=(3, 3, 1),
                                                    padding=(1, 1, 0),
                                                    bias=False),
                                          Rearrange('b h x y s -> b s (h x y)'),
                                          nn.Dropout(drop))
        self.reduce_k = nn.Conv2d(self.heads, self.heads,
                                  kernel_size=(3, 1), padding=(1, 0), stride=(4, 1),
                                  groups=self.heads, bias=False)
        self.reduce_v = nn.Conv2d(self.heads, self.heads,
                                  kernel_size=(3, 1), padding=(1, 0), stride=(4, 1),
                                  groups=self.heads, bias=False)
        self.conv_out = nn.Sequential(nn.Conv3d(in_channels=heads,
                                                out_channels=1,
                                                kernel_size=(3, 3, 1),
                                                padding=(1, 1, 0), bias=False),
                                      nn.Dropout(drop),
                                      Rearrange('b c x y s-> b c s x y'),
                                      nn.LayerNorm((patch, patch)),
                                      Rearrange('b c s x y->b c x y s')
                                      )

    def forward(self, x):
        qkv = self.conv_project(x).chunk(3, dim=-1)
        q, k, v = map(lambda a: rearrange(a, 'b s (h d) -> b h s d', h=self.heads),
                      qkv)
        k = self.reduce_k(k)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        v = self.reduce_v(v)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b c s (x y) -> b c x y s ',
                        x=self.patch, y=self.patch)
        out = self.conv_out(out)
        return out


class ConvTE(nn.Module):
    def __init__(self, heads, patch, drop):
        super().__init__()
        self.attention = Attention(heads, patch, drop)
        self.ffn = nn.Sequential(nn.Conv3d(in_channels=1, out_channels=1,
                                           kernel_size=(3, 3, 1),
                                           padding=(1, 1, 0),
                                           bias=False),
                                 nn.ReLU6(),
                                 nn.Dropout(drop)
                                 )

    def forward(self, x):
        x = x + self.attention(x)
        x = x + self.ffn(x)
        return x


class DBCT(nn.Module):
    def __init__(self, channels, patch, heads, drop, fc_dim, band_reduce):
        super().__init__()
        self.cnn_branch = CNN_branch(channels)
        self.convte_branch = nn.Sequential(nn.Conv3d(channels, 1,
                                                     kernel_size=(1, 1, 7),
                                                     padding=(0, 0, 3),
                                                     stride=(1, 1, 1)),
                                           ConvTE(heads, patch, drop)
                                           )
        self.cnn_out = nn.Sequential(nn.Conv3d(channels, channels,
                                               kernel_size=(3, 3,
                                                            band_reduce),
                                               padding=(1, 1, 0),
                                               groups=channels),
                                     nn.BatchNorm3d(channels),
                                     nn.ReLU6()
                                     )
        self.te_out = nn.Sequential(nn.Conv3d(1, channels,
                                              kernel_size=(3, 3,
                                                           band_reduce),
                                              padding=(1, 1, 0)),
                                    nn.BatchNorm3d(channels),
                                    nn.ReLU6()
                                    )
        self.out = nn.Sequential(nn.Conv3d(2 * channels, fc_dim, kernel_size=1),
                                 nn.BatchNorm3d(fc_dim),
                                 nn.ReLU6()
                                 )

    def forward(self, x):
        x_cnn = self.cnn_branch(x)
        x_te = self.convte_branch(x)
        cnn_out = self.cnn_out(x_cnn)
        te_out = self.te_out(x_te)
        out = self.out(torch.cat((cnn_out, te_out), dim=1))
        return out


class MSpeFE(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.c = channels // 4
        self.spectral1 = nn.Sequential(nn.Conv3d(self.c, self.c,
                                                 kernel_size=(1, 1, 3),
                                                 padding=(0, 0, 1),
                                                 groups=self.c),
                                       nn.BatchNorm3d(self.c),
                                       nn.ReLU6()
                                       )
        self.spectral2 = nn.Sequential(nn.Conv3d(self.c, self.c,
                                                 kernel_size=(1, 1, 7),
                                                 padding=(0, 0, 3),
                                                 groups=self.c),
                                       nn.BatchNorm3d(self.c),
                                       nn.ReLU6()
                                       )
        self.spectral3 = nn.Sequential(nn.Conv3d(self.c, self.c,
                                                 kernel_size=(1, 1, 11),
                                                 padding=(0, 0, 5),
                                                 groups=self.c),
                                       nn.BatchNorm3d(self.c),
                                       nn.ReLU6()
                                       )
        self.spectral4 = nn.Sequential(nn.Conv3d(self.c, self.c,
                                                 kernel_size=(1, 1, 15),
                                                 padding=(0, 0, 7),
                                                 groups=self.c),
                                       nn.BatchNorm3d(self.c),
                                       nn.ReLU6()
                                       )

    def forward(self, x):
        x1 = self.spectral1(x[:, 0:self.c, :])
        x2 = self.spectral2(x[:, self.c:2 * self.c, :])
        x3 = self.spectral3(x[:, 2 * self.c:3 * self.c, :])
        x4 = self.spectral4(x[:, 3 * self.c:, :])
        mspe = torch.cat((x1, x2, x3, x4), dim=1)
        return mspe


class DBCTNet(nn.Module):
    def __init__(self, channels=16, patch=9, bands=270, num_class=9,
                 fc_dim=16, heads=2, drop=0.1):
        super().__init__()
        self.band_reduce = (bands - 7) // 2 + 1
        self.stem = nn.Conv3d(1, channels, kernel_size=(1, 1, 7),
                              padding=0, stride=(1, 1, 2))
        self.mspefe = MSpeFE(channels)

        self.dbct = DBCT(channels, patch, heads, drop, fc_dim, self.band_reduce)

        self.fc = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)),
                                nn.Flatten(),
                                nn.Linear(fc_dim, num_class)
                                )

    def forward(self, x):
        # x.shape = [batch_size,1,patch_size,patch_size,spectral_bands]
        x = x.unsqueeze(1)
        b, _, _, _, _ = x.shape
        x = self.stem(x)
        x = self.mspefe(x)
        feature = self.dbct(x)
        return self.fc(feature)


class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=3, use_alpha=False, size_average=True):
        super(FocalLoss, self).__init__()
        self.class_num = class_num
        self.alpha = alpha
        self.gamma = gamma
        if use_alpha:
            # self.alpha = (torch.tensor(alpha)).view(-1,1).cuda()
            self.alpha = torch.tensor(alpha).view(-1, 1).cuda()
        self.softmax = nn.Softmax(dim=1)
        self.use_alpha = use_alpha
        self.size_average = size_average

    def forward(self, pred, target):

        prob = self.softmax(pred.view(-1, self.class_num))
        prob = prob.clamp(min=0.0001, max=1.0)  # 0.0001
        target_ = torch.zeros(target.size(0), self.class_num).cuda()
        target_.scatter_(1, target.view(-1, 1).long(), 1.)

        if self.use_alpha:
            alpha = self.alpha[target]
            # alpha = alpha.cuda()
            batch_loss = - alpha.double() * torch.pow(1 - prob, self.gamma).double() * (
                prob.log()).double() * target_.double()
        else:
            batch_loss = - torch.pow(1 - prob, self.gamma).double() * prob.log().double() * target_.double()

        batch_loss = batch_loss.sum(dim=1)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss
if __name__ == '__main__':
    model = DBCTNet(patch=16, bands=200, num_class=17)
    device = torch.device("cuda:0")
    model = model.to(device)
    model.eval()
    input = torch.randn(1,16,16,200).cuda()
    y = model(input)
    print(y.shape)

    from thop import profile
    Flops, params = profile(model, inputs=(input,))  # macs
    print('Flops: % .4fG' % (Flops / 1000000000))  # 计算量
    print('params参数量: % .4fM' % (params / 1000000))  # 参数量：等价与上面的summary输出的Total params值