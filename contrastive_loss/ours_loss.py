import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

def NT_XentLoss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N, Z = z1.shape
    device = z1.device
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
    l_pos = torch.diag(similarity_matrix, N)
    r_pos = torch.diag(similarity_matrix, -N)
    positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)
    diag = torch.eye(2 * N, dtype=torch.bool, device=device)
    diag[N:, :N] = diag[:N, N:] = diag[:N, :N]

    negatives = similarity_matrix[~diag].view(2 * N, -1)

    logits = torch.cat([positives, negatives], dim=1)
    logits /= temperature

    labels = torch.zeros(2 * N, device=device, dtype=torch.int64)

    loss = F.cross_entropy(logits, labels, reduction='sum')
    return loss / (2 * N)


class projection_MLP(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        hidden_dim = in_dim
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

# 对比
class FFTSimCLR(nn.Module):
    def __init__(self, in_chan = 256, pooling_size=4):
        super().__init__()
        self.down = nn.AdaptiveAvgPool2d((pooling_size,pooling_size)) #torch.nn.AvgPool2d(pooling_size) , nn.AdaptiveAvgPool2d((4,4))

        self.dim = int(in_chan*pooling_size**2*2)
        self.projector = projection_MLP(self.dim)
    def forward(self, x1, x2):
        x1_down = self.down(x1)
        x2_down = self.down(x2)
        # print(x1_down.shape,x2_down.shape)

        freq_x1 = torch.fft.fft2(x1_down,norm='ortho')   #.view(x1.size(0),-1) #(b,256*8*8)
        freq_x1 = torch.stack([freq_x1.real, freq_x1.imag], 1)
        freq_x1 = freq_x1.view(x1_down.size(0),-1)
        # fft_x1 = torch.abs(fft_x1)

        fft_x2 = torch.fft.fft2(x2_down,norm='ortho')    #.view(x2.size(0),-1)
        freq_x2 = torch.stack([fft_x2.real, fft_x2.imag], 1)
        freq_x2 = freq_x2.view(x2_down.size(0),-1)
        # fft_x2 = torch.abs(fft_x2)

        z1 = self.projector(freq_x1)
        z2 = self.projector(freq_x2)

        loss = NT_XentLoss(z1, z2)
        return loss

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

if __name__ == "__main__":
    x1 = torch.rand(10,256,16,16)
    x2 = torch.rand(10,256,16,16)
    loss = FFTSimCLR()
    y = loss(x1,x2)
    print(y)
