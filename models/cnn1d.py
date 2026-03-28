import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init
import math
class CNN1D(nn.Module):
    """
    Deep Convolutional Neural Networks for Hyperspectral Image Classification
    Wei Hu, Yangyu Huang, Li Wei, Fan Zhang and Hengchao Li
    Journal of Sensors, Volume 2015 (2015)
    https://www.hindawi.com/journals/js/2015/258619/
    """

    @staticmethod
    def weight_init(m):
        # [All the trainable parameters in our CNN should be initialized to
        # be a random value between −0.05 and 0.05.]
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            init.uniform_(m.weight, -0.05, 0.05)
            init.zeros_(m.bias)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_channels)
            x = self.pool(self.conv(x))
        return x.numel()

    def __init__(self, in_channels, n_classes, patch_size=8, kernel_size=None, pool_size=None):
        super(CNN1D, self).__init__()
        self.patch_size = patch_size
        if kernel_size is None:
            # [In our experiments, k1 is better to be [ceil](n1/9)]
            kernel_size = math.ceil(in_channels / 9)
        if pool_size is None:
            # The authors recommand that k2's value is chosen so that the pooled features have 30~40 values
            # ceil(kernel_size/5) gives the same values as in the paper so let's assume it's okay
            pool_size = math.ceil(kernel_size / 5)
        self.input_channels = in_channels

        # [The first hidden convolution layer C1 filters the n1 x 1 input data with 20 kernels of size k1 x 1]
        self.conv = nn.Conv1d(1, 20, kernel_size)
        self.pool = nn.MaxPool1d(pool_size)
        self.features_size = self._get_final_flattened_size()
        # [n4 is set to be 100]
        self.fc1 = nn.Linear(self.features_size, 100)
        self.fc2 = nn.Linear(100, n_classes)
        self.apply(self.weight_init)

    def forward(self, x):
        # [In our design architecture, we choose the hyperbolic tangent function tanh(u)]
        x = x[:,:,self.patch_size//2+1,self.patch_size//2+1]
        x = x.squeeze(dim=-1).squeeze(dim=-1)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = torch.tanh(self.pool(x))
        x = x.view(-1, self.features_size)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

def cnn1d(dataset, patch_size):
    model = None
    if dataset == 'Salinas':
        model = CNN1D(in_channels=204,n_classes=17,patch_size=patch_size)
    elif dataset == 'IndianPines':
        model = CNN1D(in_channels=200, n_classes=17,patch_size=patch_size)
    elif dataset == 'whulk':
        model = CNN1D(in_channels=270, n_classes=9,patch_size=patch_size)
    elif dataset == 'botswana':
        model = CNN1D(in_channels=145, n_classes=14,patch_size=patch_size)
    elif dataset == 'flt':
        model = CNN1D(in_channels=80, n_classes=10,patch_size=patch_size)
    elif dataset == 'WHUHC':
        model = CNN1D(in_channels=274, n_classes=17,patch_size=patch_size)
    elif dataset == 'KSC':
        model = CNN1D(in_channels=176, n_classes=14,patch_size=patch_size)
    return model

if __name__ == '__main__':
    from thop import profile
    P = 8
    t = torch.randn(size=(1, 200, P, P))
    print("input shape:", t.shape)
    net = cnn1d(dataset='IndianPines', patch_size=P)
    print("output shape:", net(t).shape)
    flops, params = profile(net, inputs=(t,))
    print('params', params)
    print('flops', flops)  ## 打印计算量