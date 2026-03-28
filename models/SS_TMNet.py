import torch
import torch.nn as nn
import numpy as np
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
import torch.nn.functional as F
from torch.nn import init

class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h,dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        # print(q.shape)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)
        # print(q.shape)
        # print(att.shape)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class SimplifiedScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, h,dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SimplifiedScaledDotProductAttention, self).__init__()

        self.d_model = d_model
        self.d_k = d_model//h
        self.d_v = d_model//h
        self.h = h

        self.fc_o = nn.Linear(h * self.d_v, d_model)
        self.dropout=nn.Dropout(dropout)



        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = queries.view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = keys.view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = values.view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class AttentionModule(nn.Module):

    def __init__(self, d_model=512, kernel_size=3, H=7, W=7):
        super().__init__()
        self.position_attention_module_w = SpatialAttentionModule(d_model=d_model, kernel_size=(1, 3), stride=1,
                                                                  padding=(0, 1), H=H, W=W)
        self.position_attention_module_h = SpatialAttentionModule(d_model=d_model, kernel_size=(3, 1), stride=1,
                                                                  padding=(1, 0), H=H, W=W)
        self.channel_attention_module = SpectralAttentionModule(d_model=d_model, kernel_size=3, H=H, W=W)
        self.reweight = Mlp(d_model, d_model // 4, d_model * 3)

    def forward(self, input):
        p_out_w = self.position_attention_module_w(input)
        p_out_h = self.position_attention_module_h(input)
        c_out = self.channel_attention_module(input)
        return p_out_w + p_out_h + c_out

class SpatialAttentionModule(nn.Module):

    def __init__(self, d_model=512, kernel_size=3, stride=None, padding=None, H=7, W=7):
        super().__init__()
        self.cnn2 = nn.Conv2d(d_model, d_model, kernel_size=kernel_size, stride=stride, padding=padding, dilation=1,
                              groups=d_model, bias=True)
        self.pa = ScaledDotProductAttention(d_model, d_k=d_model, d_v=d_model, h=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        bs, c, h, w = x.shape
        y = self.cnn2(x)
        y = y.view(bs, c, -1).permute(0, 2, 1)
        y = self.pa(y, y, y)
        y = y.permute(0, 2, 1).view(bs, c, h, w)
        y = self.gamma * y + x
        return y


class SpectralAttentionModule(nn.Module):

    def __init__(self, d_model=512, kernel_size=3, H=7, W=7):
        super().__init__()
        self.pa = SimplifiedScaledDotProductAttention(H * W, h=1)
        self.cnn2 = nn.Conv2d(d_model, d_model, kernel_size=1, bias=True)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        bs, c, h, w = x.shape
        y = self.cnn2(x)
        y = y.view(bs, c, -1)
        y = self.pa(y, y, y)
        y = y.view(bs, c, h, w)
        y = self.gamma * y + x
        return y


class SSAM(nn.Module):
    def __init__(self, dim, segment_dim=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.segment_dim = segment_dim
        self.ssam=AttentionModule(d_model=dim,kernel_size=3,H=segment_dim,W=segment_dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        add = x
        c = x.permute(0, 3, 1, 2)
        c = self.ssam(c)
        c = c.permute(0, 2, 3, 1)
        x = self.proj(c)
        x = x+add
        x = self.proj_drop(x)

        return x

class MSCP(nn.Module):
    """ Image to Patch Embedding
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

    def __init__(self, img_size=15, patch_size=3, in_chans=3, embed_dim=16):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 4, (11, 3, 3), stride=(2, 2, 2), padding=1)
        self.conv2_1 = nn.Conv3d(4, 4, (1, 1, 1), padding=(0, 0, 0))
        self.conv2_2 = nn.Conv3d(4, 4, (3, 1, 1), padding=(1, 0, 0))
        self.conv2_3 = nn.Conv3d(4, 4, (5, 1, 1), padding=(2, 0, 0))
        self.conv2_4 = nn.Conv3d(4, 4, (11, 1, 1), padding=(5, 0, 0))
        self.bn1 = nn.BatchNorm3d(4)
        self.conv_add = nn.Conv3d(4, 8, (9, 3, 3), stride=(2, 1, 1), padding=1)
        self.conv3_1 = nn.Conv3d(8, 8, (1, 1, 1), padding=(0, 0, 0))
        self.conv3_2 = nn.Conv3d(8, 8, (3, 1, 1), padding=(1, 0, 0))
        self.conv3_3 = nn.Conv3d(8, 8, (5, 1, 1), padding=(2, 0, 0))
        self.conv3_4 = nn.Conv3d(8, 8, (11, 1, 1), padding=(5, 0, 0))

        self.conv3_5 = nn.Conv3d(8, 8, (1, 1, 1), padding=(0, 0, 0))
        self.conv3_6 = nn.Conv3d(8, 8, (3, 1, 1), padding=(1, 0, 0))
        self.conv3_7 = nn.Conv3d(8, 8, (5, 1, 1), padding=(2, 0, 0))
        self.conv3_8 = nn.Conv3d(8, 8, (11, 1, 1), padding=(5, 0, 0))

        self.bn2 = nn.BatchNorm3d(8)
        self.conv4 = nn.Conv3d(8, 8, (1, 1, 1),stride=(1, 1, 1))
        self.apply(self.weight_init)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x2_1 = self.conv2_1(x)
        x2_2 = self.conv2_2(x)
        x2_3 = self.conv2_3(x)
        x2_4 = self.conv2_4(x)
        x = x2_1 + x2_2 + x2_3 + x2_4 +x
        x= self.bn1(x)
        x = F.relu(x)
        x = F.relu(self.conv_add(x))
        x3_1 = self.conv3_1(x)
        x3_2 = self.conv3_2(x)
        x3_3 = self.conv3_3(x)
        x3_4 = self.conv3_4(x)
        x = x3_1 + x3_2 + x3_3 + x3_4 + x
        x = F.relu(x)
        x3_5 = self.conv3_5(x)
        x3_6 = self.conv3_6(x)
        x3_7 = self.conv3_7(x)
        x3_8 = self.conv3_8(x)
        x = x3_5 + x3_6 + x3_7 + x3_8 + x
        x= self.bn2(x)
        x = F.relu(x)
        x = F.gelu(self.conv4(x))
        B, D, H, W, C = x.shape
        x = x.reshape(B, D*H, W, C)
        return x

class PermutatorBlock(nn.Module):

    def __init__(self, dim, segment_dim, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip_lam=1.0, mlp_fn=SSAM):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = mlp_fn(dim, segment_dim=segment_dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)#MLP2
        self.skip_lam = skip_lam

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        return x


class Downsample(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, in_embed_dim, out_embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)  # B, C, H, W
        x = x.permute(0, 2, 3, 1)
        return x


def basic_blocks(dim, index, layers, segment_dim, mlp_ratio=3., qkv_bias=True, qk_scale=None, \
                 attn_drop=0, drop_path_rate=0., skip_lam=1.0, mlp_fn=SSAM, **kwargs):
    blocks = []

    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(PermutatorBlock(dim, segment_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, \
                                      attn_drop=attn_drop, drop_path=block_dpr, skip_lam=skip_lam, mlp_fn=mlp_fn))

    blocks = nn.Sequential(*blocks)

    return blocks


class SS_TMNet(nn.Module):

    def __init__(self, layers, img_size=15, patch_size=3, in_chans=3, num_classes=1000,
                 embed_dims=None, transitions=None, segment_dim=None, mlp_ratios=None, skip_lam=1.0,
                 qkv_bias=False, qk_scale=None, drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, mlp_fn=SSAM):

        super().__init__()
        self.num_classes = num_classes

        self.patch_embed = MSCP(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                      embed_dim=embed_dims[0])
        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers, segment_dim[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                                 qk_scale=qk_scale, attn_drop=attn_drop_rate, drop_path_rate=drop_path_rate,
                                 norm_layer=norm_layer, skip_lam=skip_lam,
                                 mlp_fn=mlp_fn)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if transitions[i] or embed_dims[i] != embed_dims[i + 1]:
                patch_size = 2 if transitions[i] else 1
                network.append(Downsample(embed_dims[i], embed_dims[i + 1], patch_size))
        self.network = nn.ModuleList(network)

        self.norm = norm_layer(embed_dims[-1])

        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_cls_head = nn.Linear(368, num_classes)
        self.down_sample = Downsample(embed_dims[0], 512, 2)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        x = x.permute(0, 2, 3, 1)
        return x

    def forward_tokens(self, x):
        x_embedding = x
        for idx, block in enumerate(self.network):
            x = block(x)

        B, H, W, C = x.shape
        x_embedding = self.down_sample(x_embedding)
        x = x +x_embedding

        x = x.reshape(B, -1, C)
        return x

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.forward_embeddings(x)
        x = self.forward_tokens(x)
        x = self.norm(x)
        return self.head(x.mean(1))

def SSTMNet(dataset, patchsize):
    model = None
    transitions = [False, True, False, False]
    segment_dim = [8, 8, 4, 4]
    mlp_ratios = [3, 3, 3, 3]
    layers = [8, 5, 8, 5]
    if dataset == 'Salinas':
        embed_dims = [368, 368, 512, 512]
        model = SS_TMNet(layers=layers, img_size=patchsize, in_chans=204, num_classes=17, embed_dims=embed_dims,
                     patch_size=3, transitions=transitions,
                     segment_dim=segment_dim, mlp_ratios=mlp_ratios, mlp_fn=SSAM)
    elif dataset == 'WHUHC':
        embed_dims = [512, 512, 512, 512]
        model = SS_TMNet(layers=layers, img_size=patchsize, in_chans=274, num_classes=17, embed_dims=embed_dims,
                     patch_size=3, transitions=transitions,
                     segment_dim=segment_dim, mlp_ratios=mlp_ratios, mlp_fn=SSAM)
    elif dataset == 'KSC':
        embed_dims = [312, 312, 512, 512]
        model = SS_TMNet(layers=layers, img_size=patchsize, in_chans=176, num_classes=14, embed_dims=embed_dims,
                     patch_size=3, transitions=transitions,
                     segment_dim=segment_dim, mlp_ratios=mlp_ratios, mlp_fn=SSAM)
    elif dataset == 'IndianPines':
        embed_dims = [360, 360, 512, 512]
        model = SS_TMNet(layers=layers, img_size=patchsize, in_chans=200, num_classes=17, embed_dims=embed_dims,
                     patch_size=3, transitions=transitions,
                     segment_dim=segment_dim, mlp_ratios=mlp_ratios, mlp_fn=SSAM)
    return model


if __name__ == '__main__':
    # model = DCTN(layers=[2, 2, 5, 3], img_size=16, in_chans=200, num_classes=17, embed_dims=[440, 440, 512, 512],
    #              patch_size=3, transitions=[False, True, False, False],
    #              segment_dim=[8, 8, 4, 4], mlp_ratios=[3, 3, 3, 3], mlp_fn=EISA, dateset="IndianPines")
    model = SSTMNet("WHUHC", 16)
    device = torch.device("cuda:0")
    model = model.to(device)
    model.eval()
    input = torch.randn(1,274, 16, 16).cuda()
    y = model(input)
    print(y.shape)

    from thop import profile

    Flops, params = profile(model, inputs=(input,))  # macs
    print('Flops: % .4fG' % (Flops / 1000000000))  # 计算量
    print('params参数量: % .4fM' % (params / 1000000))  # 参数量：等价与上面的summary输出的Total params值