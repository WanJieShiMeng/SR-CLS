import numpy as np
import torch.utils.data as data
import scipy.io as sio
import torch
from torch.nn.functional import interpolate
import h5py
from matlab_functions import imresize

def data_augmentation(label, mode=0):
    if mode == 0:
        # original
        return label
    elif mode == 1:
        # flip up and down
        return np.flipud(label)   # flipud  上下翻转
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(label)    # rot90  逆时针旋转90度*k, k = 负数时顺时针旋转
    elif mode == 3:
        # rotate 90 degree and flip up and down
        return np.flipud(np.rot90(label))
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(label, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        return np.flipud(np.rot90(label, k=2))
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(label, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        return np.flipud(np.rot90(label, k=3))

def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in [".mat"])


def get_patches_by_stride_split(X,patch_size,stride):
    h, w, c = X.shape
    row = (h - patch_size) // stride + 1
    col = (w - patch_size) // stride + 1
    newX = X
    res = np.zeros((row * col, patch_size, patch_size, c))
    index = 0
    for i in range(row):
        for j in range(col):
            res[index, :, :, :] = newX[i * stride:i * stride + patch_size, j * stride:j * stride + patch_size, :]
            index += 1
    return res


def split_dataset(data_name,patch_size,stride):
    eval_p = 0.1
    if data_name == "Chikusei":
        ori_data = h5py.File('/home/wxy/Chikusei/HyperspecVNIR_Chikusei_20140729.mat')['chikusei']
        ori_data = np.transpose(ori_data, (2, 1, 0))
        ori_data = ori_data[106:2410,143:2191,:]
        normal_data = ori_data / np.max(ori_data)
        test_data = []
        for i in range(ori_data.shape[1] // 512):
            test_data.append(normal_data[:512,i*512:(i+1)*512,:])
        res_data = normal_data[512:, :, :]

    elif data_name == "Pavia":
        ori_data = sio.loadmat('/home/wxy/Pavia/Pavia.mat')['pavia']
        normal_data = ori_data / np.max(ori_data)
        test_data = []
        for i in range(8):
            test_data.append(normal_data[i * 128:(i + 1) * 128, :128, :])
        res_data = normal_data[:, 128:, :]

    elif data_name == "houstonU":
        ori_data = h5py.File('/home/wxy/houstonU/HoustonU.mat')['houstonU']
        ori_data = np.transpose(ori_data, (2, 1, 0))[:, :, :-2]
        normal_data = ori_data / np.max(ori_data)
        test_data = []
        for i in range(4):
            test_data.append(normal_data[i * 128:(i + 1) * 128, :128, :])
        res_data = normal_data[:, 128:, :]

    test_data = np.array(test_data)
    # 最后的划分为eval集
    res_patch_data = get_patches_by_stride_split(res_data, patch_size, stride)
    eval_num = int(res_patch_data.shape[0] * eval_p)
    train_num = res_patch_data.shape[0] - eval_num
    train_data = res_patch_data[:train_num,:,:,:]
    eval_data = res_patch_data[train_num:,:,:,:]

    return train_data.transpose((0,3,1,2)),eval_data.transpose((0,3,1,2)),test_data.transpose((0,3,1,2))


class HSTrainingData(data.Dataset):
    def __init__(self, splited_data, scale, augment=True, use_3D=False):
        self.data = torch.tensor(splited_data)
        self.scale = scale

        self.augment = augment
        self.use_3DConv = use_3D
        if self.augment:
            self.factor = 8
        else:
            self.factor = 1

    def __getitem__(self, index):
        aug_num = 0
        if self.augment:
            index = index // self.factor
            aug_num = int(index % self.factor)
        gt = self.data[index]
        ms = imresize(gt,1/self.scale)
        lms = imresize(ms,self.scale)
        gt = gt.permute(1, 2, 0).numpy()
        ms = ms.permute(1,2,0).numpy()
        lms = lms.permute(1,2,0).numpy()

        ms, lms, gt = data_augmentation(ms, mode=aug_num), data_augmentation(lms, mode=aug_num), \
                      data_augmentation(gt, mode=aug_num)

        if self.use_3DConv:
            ms, lms, gt = ms[np.newaxis, :, :, :], lms[np.newaxis, :, :, :], gt[np.newaxis, :, :, :]
            ms = torch.from_numpy(ms.copy()).permute(0, 3, 1, 2)
            lms = torch.from_numpy(lms.copy()).permute(0, 3, 1, 2)
            gt = torch.from_numpy(gt.copy()).permute(0, 3, 1, 2)
        else:
            ms = torch.from_numpy(ms.copy()).permute(2, 0, 1)
            lms = torch.from_numpy(lms.copy()).permute(2, 0, 1)
            gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        return ms, lms, gt

    def __len__(self):
        return self.data.shape[0]*self.factor


class HSTestData(data.Dataset):
    def __init__(self, splited_data,scale):
        self.data = torch.tensor(splited_data)
        self.scale = scale

    def __getitem__(self, index):
        gt = self.data[index]
        ms = imresize(gt,1/self.scale)
        lms = imresize(ms,self.scale)
        return ms, lms, gt

    def __len__(self):
        return self.data.shape[0]


if __name__ == "__main__":
    data_name = "Chikusei"
    scale = 4
    patch_size = 0
    stride = 0
    if data_name == "Chikusei":
        if scale == 2:
            patch_size = 64
            stride = 32
        elif scale == 4:
            patch_size = 128
            stride = 64
        elif scale == 8:
            patch_size = 256
            stride = 128
    elif data_name == "Pavia" or data_name == "houstonU":
        if scale == 2:
            patch_size = 64
            stride = 32
        elif scale == 4:
            patch_size = 128
            stride = 64
        elif scale == 8:
            patch_size = 128
            stride = 64
    train_data, eval_data, test_data = split_dataset(data_name, patch_size, stride)
    print(train_data.shape, eval_data.shape, test_data.shape)


    from skimage.measure import compare_psnr
    # from scipy.misc import imresize
    def compare_mpsnr(x_true, x_pred, data_range):
        """
        :param x_true: Input image must have three dimension (H, W, C)
        :param x_pred:
        :return:
        """
        x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
        channels = x_true.shape[2]
        total_psnr = [compare_psnr(x_true[:, :, k], x_pred[:, :, k], data_range=data_range)
                      for k in range(channels)]
        return np.mean(total_psnr)
    train_dataloader = HSTrainingData(train_data,scale)
    ms,lms,gt = train_dataloader[1]
    print(ms.shape,lms.shape,gt.shape)
    test_dataloader = HSTestData(test_data,scale)
    # --------------------------------------------------------------------------------------------------------------- #
    # 测试imsize
    psnr = 0
    for i in range(len(test_dataloader)):
        ms,lms,gt = test_dataloader[i]
        gt = gt.numpy().transpose(1, 2, 0)
        lms = lms.numpy().transpose(1, 2, 0)
        # lms = np.zeros(gt.shape,dtype=np.float32)
        # for i in range(ms.shape[0]):
        #     lms[:,:,i] = imresize(ms[i,:,:],(gt.shape[0],gt.shape[1]),'bicubic',mode='F')
        psnr += compare_mpsnr(gt,lms,1.)
        # print('=========')
        # print(gt[0,0,:3])
    print(psnr / len(test_dataloader))
    # --------------------------------------------------------------------------------------------------------------- #
    # _,lms,gt = test_dataloader[2]
    # rgb = [70, 100, 36]
    # gt = gt[rgb,:,:]
    # lms = lms[rgb, :, :]
    #
    # # gt = gt.numpy().transpose(1, 2, 0)
    # # lms = lms.numpy().transpose(1, 2, 0)
    #
    # from torchvision import transforms
    # import matplotlib.pyplot as plt
    # def trans(x):
    #     if type(x) == np.ndarray:
    #         return np.transpose(x, (1, 2, 0))
    #     else:
    #         return x.permute(1, 2, 0)
    #
    #
    # def totype(x):
    #     if type(x) == np.ndarray:
    #         return x.astype(np.uint8)
    #     else:
    #         return x.numpy().astype(np.uint8)
    # reverse_transforms = transforms.Compose([
    #     # transforms.Lambda(lambda t: (t + 1) / 2),
    #     transforms.Lambda(lambda t: trans(t)),  # CHW to HWC
    #     transforms.Lambda(lambda t: t * 255.),
    #     transforms.Lambda(lambda t: totype(t)),
    #     # transforms.ToPILImage(),
    # ])
    # gt = reverse_transforms(gt)
    # plt.imshow(gt)
    #
    # lms = reverse_transforms(lms)
    # plt.imshow(lms)
    #
    # plt.show()

