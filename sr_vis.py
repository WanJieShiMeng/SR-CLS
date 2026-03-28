import torch
import matplotlib.pyplot as plt
from models import OursSwinCNNFuse,MultiTask
from dataset.dataset import get_dataset,HyperX
import numpy as np
from utils import sample_gt
import torch.nn.functional as F
import torch.utils.data as data
from utils import set_random_seed
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# set_random_seed(202401)
hyper = {'patch_size': 64,
         'n_bands': 200,
         'n_classes': 17,
         'dataset': 'IndianPines',
         'ignored_labels': [0],
         'flip_augmentation': False,
         'radiation_augmentation': False,
         'mixture_augmentation': False,
         'center_pixel': True,
         'supervision': 'full'}
multi_model = MultiTask(Swin=False, fusion=True, cross=True, contrast=True, **hyper)
multi_model.load_state_dict(torch.load('/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/IndianPines/10_label/bestModel_epoch65_0.8313.pth',map_location='cuda:0'))


img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset('IndianPines', "./Datasets/", use_PCA=False)
# gt标签mask边缘部分
gt_copy = np.zeros_like(gt)
half_patch = int(np.floor(16 / 2))
gt_copy[half_patch:-(half_patch - 1), half_patch:-(half_patch - 1)] = gt[half_patch:-(half_patch - 1),
                                                                      half_patch:-(half_patch - 1)]
gt = gt_copy
# ----------------------------------------------------------------------------------------------- #

train_gt, test_gt = sample_gt(gt, 10, mode="fixed")
val_gt, test_gt = sample_gt(test_gt, 0.05, mode="random")
train_dataset = HyperX(img, test_gt, **hyper)
train_loader = data.DataLoader(
    train_dataset,
    batch_size=64,
    # pin_memory=hyperparams['device'],
    shuffle=True,
)

# --------------------------------------------------------------------------------------------------------------------- #
# draw
from torchvision import transforms
def trans_img(x):
    def trans(x):
        if type(x) == np.ndarray:
            return np.transpose(x, (1, 2, 0))
        else:
            return x.permute(1, 2, 0)

    def totype(x):
        if type(x) == np.ndarray:
            return x.astype(np.uint8)
        else:
            return x.numpy().astype(np.uint8)

    reverse_transforms = transforms.Compose([
        # transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: trans(t)),  # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: totype(t)),
        # transforms.ToPILImage(),
    ])

    return reverse_transforms(x)

rgb = [43, 21, 11]
device = torch.device("cuda")
multi_model.to(device)

data = train_dataset[0][0].unsqueeze(0)
data= data.to(device)
output_sr= multi_model.forward_sr(F.interpolate(data, size=[i * 2 for i in data.size()[2:]], mode='bilinear', align_corners=True))
output_sr = output_sr.detach()

ori_data = data[0][rgb,:,:].cpu()
plt.subplot(131)
plt.imshow(trans_img(ori_data))
plt.axis('off')

lr_x = F.interpolate(data, size=[i // 2 for i in data.size()[2:]], mode='bilinear', align_corners=True)
bic_x = F.interpolate(lr_x, size=[i * 2 for i in lr_x.size()[2:]], mode='bilinear', align_corners=True)
bic_data = bic_x[0][rgb,:,:].cpu()
plt.subplot(132)
plt.imshow(trans_img(bic_data))
plt.axis('off')

output_sr = output_sr[0][rgb,:,:].cpu()
plt.subplot(133)
plt.imshow(trans_img(output_sr))
plt.axis('off')

plt.show()