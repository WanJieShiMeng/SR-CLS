import torch
import datetime
import numpy as np
import joblib
from tqdm import tqdm
import torch.optim as optim
import copy
import os
from models.MCTGCL.GCN_model import *
import models.MCTGCL.supervised_contrastive_loss as supervised_contrastive_loss

from sklearn.neighbors import kneighbors_graph
def aff_to_adj(last_layer_data_src):
    last_layer_data_src = F.normalize(last_layer_data_src, dim=-1)
    features1 = last_layer_data_src.cpu().detach().numpy()

    adj_nei = kneighbors_graph(features1, 10, mode='distance')
    adj_nei = adj_nei.A
    sigam=1
    for i in range(adj_nei.shape[0]):
        for j in range(adj_nei.shape[1]):
            if adj_nei[i][j] != 0:
                adj_nei[i][j] = np.exp(-adj_nei[i][j]/(sigam*sigam))
    adj_d = np.sum(adj_nei,axis=1, keepdims=True)
    adj_d = np.diag(np.squeeze(adj_d**(-0.5)))
    adj_w = np.matmul(adj_nei,adj_d)
    adj_w = np.matmul(adj_d,adj_w)
    adj_nei = adj_w+np.eye(adj_w.shape[0])
    adj_nei = torch.from_numpy(adj_nei).cuda(1).to(torch.float32)
    return adj_nei



def train_MCTGCL(
        net,
        optimizer,
        criterion,
        train_loader,
        epoch,
        scheduler=None,
        display_iter=20,
        device=torch.device("cpu"),
        display=None,
        val_loader=None,
        supervision="full",
        name="name",
        save_dir=None,
        data_labeled_loader = None,
        num_class = 17
):
    """
    Training loop to optimize a network for several epochs and a specified loss

    Args:
        net: a PyTorch model
        optimizer: a PyTorch optimizer
        train_loader: a PyTorch dataset loader
        epoch: int specifying the number of training epochs
        criterion: a PyTorch-compatible loss function, e.g. nn.CrossEntropyLoss
        device (optional): torch device to use (defaults to CPU)
        display_iter (optional): number of iterations before refreshing the
        display (False/None to switch off).
        scheduler (optional): PyTorch scheduler
        val_loader (optional): validation dataset
        supervision (optional): 'full' or 'semi'
    """

    if criterion is None:
        raise Exception("Missing criterion. You must specify a loss function.")
    # -------------------------------------------------------------------------------------------------------------- #
    # 保留最好的一代
    best_model = None
    best_val_acc = 0.0
    best_epoch = None
    # -------------------------------------------------------------------------------------------------------------- #
    net.to(device)

    save_epoch = 1  # epoch // 20 if epoch > 20 else 1

    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    iter_ = 1
    loss_win, val_win = None, None
    val_accuracies = []

    src_gcn_module = GCN_M(nfeat=128,
                           nhid=128,
                           nclass=1,
                           dropout=0.3).to(device)
    src_optim = optim.Adam(src_gcn_module.parameters(), lr=0.001)

    for e in tqdm(range(1, epoch + 1), desc="Training the network"):
        # Set the network to training mode
        net.train()
        avg_loss = 0.0

        data_labeled = next(iter(data_labeled_loader))
        data_all, target_all = data_labeled[0].to(device), data_labeled[1]

        data_all_aug = torch.flip(data_all.clone().permute(0, 1, 3, 2), dims=[2])

        # Run the training loop for one epoch
        for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)

            # 通过输入得到预测的输出
            outputs, _ = net(data)
            # GCN_model
            # 计算每个类的均值
            outputs_all, features_all = net(data_all)
            src_adj_nei = aff_to_adj(features_all).to(device)
            outputs_src = src_gcn_module(features_all, src_adj_nei)
            outputs_src = F.normalize(outputs_src, dim=-1)

            outputs_all_aug, features_all_aug = net(data_all_aug)
            tar_adj_nei = aff_to_adj(features_all_aug).to(device)
            outputs_tar = src_gcn_module(features_all_aug, tar_adj_nei)
            outputs_tar = F.normalize(outputs_tar, dim=-1)

            contrastiveLoss = supervised_contrastive_loss.SupConLoss(1)  # 1 is the temperature parameter
            # 遍历查找每个数值 i 的位置
            out_class_list = []
            out_class2_list = []
            for i in range(num_class):  # i 从 0 到 8  # 9
                indices = torch.nonzero(target_all == i).squeeze()  # 使用 torch.nonzero() 找到所有等于 i 的索引
                out_class = outputs_tar[indices, :].mean(dim=0)
                out_class2 = outputs_src[indices, :].mean(dim=0)
                out_class_list.append(out_class)
                out_class2_list.append(out_class2)
            out_class_tensor = torch.stack(out_class_list, dim=0)
            out_class2_tensor = torch.stack(out_class2_list, dim=0)
            features_class = torch.cat((out_class_tensor, out_class2_tensor), dim=0)
            features_class = features_class.unsqueeze(1)
            labels = torch.arange(num_class)  # 9
            labels_class = torch.cat((labels, labels), dim=0)
            f_contrastive_loss = contrastiveLoss(features_class, labels_class)
            # print(f_contrastive_loss)
            # 计算损失函数
            loss = criterion(outputs, target) + 0.5 * f_contrastive_loss  # a
            # 优化器梯度归零
            optimizer.zero_grad()
            src_optim.zero_grad()
            # 反向传播
            loss.backward()
            optimizer.step()
            src_optim.step()

            avg_loss += loss.item()
            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100): iter_ + 1])

            if display_iter and iter_ % display_iter == 0:
                string = "Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}"
                string = string.format(
                    e,
                    epoch,
                    batch_idx * len(data),
                    len(data) * len(train_loader),
                    100.0 * batch_idx / len(train_loader),
                    mean_losses[iter_],
                )
                update = None if loss_win is None else "append"
                loss_win = display.line(
                    X=np.arange(iter_ - display_iter, iter_),
                    Y=mean_losses[iter_ - display_iter: iter_],
                    win=loss_win,
                    update=update,
                    opts={
                        "title": "Training loss",
                        "xlabel": "Iterations",
                        "ylabel": "Loss",
                    },
                )
                tqdm.write(string)

                if len(val_accuracies) > 0:
                    val_win = display.line(
                        Y=np.array(val_accuracies),
                        X=np.arange(len(val_accuracies)),
                        win=val_win,
                        opts={
                            "title": "Validation accuracy",
                            "xlabel": "Epochs",
                            "ylabel": "Accuracy",
                        },
                    )
            iter_ += 1
            del (data, target, loss)

        # Update the scheduler
        avg_loss /= len(train_loader)
        if val_loader is not None:
            val_acc = val_MCTGCL(net, val_loader, device=device)
            val_accuracies.append(val_acc)
            metric = -val_acc
        else:
            metric = avg_loss

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)
        elif scheduler is not None:
            scheduler.step()

        # -------------------------------------------------------------------------------------------------------------- #
        # 保留最好的一代
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = e
            best_model = copy.deepcopy(net)
        # -------------------------------------------------------------------------------------------------------------- #
    # best_model = copy.deepcopy(net)
    save_model(
        best_model,
        name,
        train_loader.dataset.name,
        epoch=best_epoch,
        metric=abs(best_val_acc),
        save_dir=save_dir
    )
    return best_epoch, best_val_acc, best_model


def val_MCTGCL(net, data_loader, device="cpu"):
    net.eval()
    # TODO : fix me using metrics()
    accuracy, total = 0.0, 0.0
    ignored_labels = data_loader.dataset.ignored_labels
    for batch_idx, (data, target) in enumerate(data_loader):
        with torch.no_grad():
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)

            output,_ = net(data)
            _, output = torch.max(output, dim=1)
            for out, pred in zip(output.view(-1), target.view(-1)):
                if pred.item() in ignored_labels:
                    continue
                else:
                    accuracy += out.item() == pred.item()
                    total += 1
    net.train()
    return accuracy / total


from utils import grouper, sliding_window, count_sliding_window


def test_MCTGCL(net, img, hyperparams):
    """
    Test a model on a specific image
    """
    net.eval()
    patch_size = hyperparams["patch_size"]
    center_pixel = hyperparams["center_pixel"]
    batch_size, device = 128, hyperparams["device"]  # 500
    n_classes = hyperparams["n_classes"]

    kwargs = {
        "step": hyperparams["test_stride"],
        "window_size": (patch_size, patch_size),
    }
    probs = np.zeros(img.shape[:2] + (n_classes,))

    iterations = count_sliding_window(img, **kwargs) // batch_size
    for batch in tqdm(
            grouper(batch_size, sliding_window(img, **kwargs)),
            total=(iterations),
            desc="Inference on the image",
    ):
        with torch.no_grad():
            if patch_size == 1:
                data = [b[0][0, 0] for b in batch]
                data = np.copy(data)
                data = torch.from_numpy(data)
            else:
                data = [b[0] for b in batch]
                data = np.copy(data)
                # ========================================= #
                # temp_data = np.zeros((data.shape[0], data.shape[1] // 2, data.shape[2] // 2, data.shape[3]), dtype="float32")
                # for i in range(data.shape[0]):
                #     temp_data[i] = degradation(data[i])
                # data = temp_data.transpose(0, 3, 1, 2)
                # ========================================= #
                data = data.transpose(0, 3, 1, 2)
                data = torch.from_numpy(data)
                # ========================================= #
                # data = F.interpolate(data, size=(data.shape[1] // 2, data.shape[2] // 2), mode="bilinear")
                # noise = torch.from_numpy(np.random.normal(0, 10 / 255.0, data.shape).astype(np.float32))
                # data = data + noise
                # data = torch.clip(data, 0.0, 1.0)
                # ========================================= #
                # data = data.unsqueeze(1)

            indices = [b[1:] for b in batch]
            data = data.to(device)
            output, _ = net(data)
            if isinstance(output, tuple):
                output = output[0]
            output = output.to("cpu")

            if patch_size == 1 or center_pixel:
                output = output.numpy()
            else:
                output = np.transpose(output.numpy(), (0, 2, 3, 1))
            for (x, y, w, h), out in zip(indices, output):
                if center_pixel:
                    probs[x + w // 2, y + h // 2] += out
                else:
                    probs[x: x + w, y: y + h] += out
    return probs


def save_model(model, model_name, dataset_name, **kwargs):
    model_dir = kwargs['save_dir']
    """
    Using strftime in case it triggers exceptions on windows 10 system
    """
    time_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    if isinstance(model, torch.nn.Module):
        # time_str
        filename = "bestModel" + "_epoch{epoch}_{metric:.4f}".format(
            **kwargs
        )
        tqdm.write("Saving neural network weights in {}".format(filename))
        torch.save(model.state_dict(), model_dir + filename + ".pth")
    else:
        filename = "bestModel"
        tqdm.write("Saving model params in {}".format(filename))
        joblib.dump(model, model_dir + filename + ".pkl")

import torch.utils.data
class HyperX_MCTGCL(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX_MCTGCL, self).__init__()
        self.data = data
        self.label = gt
        self.name = hyperparams["dataset"]
        self.patch_size = hyperparams["patch_size"]
        self.ignored_labels = set(hyperparams["ignored_labels"])
        self.flip_augmentation = hyperparams["flip_augmentation"]
        self.radiation_augmentation = hyperparams["radiation_augmentation"]
        self.mixture_augmentation = hyperparams["mixture_augmentation"]
        self.center_pixel = hyperparams["center_pixel"]
        supervision = hyperparams["supervision"]
        # Fully supervised : use all pixels with label not ignored
        if supervision == "full":
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        # Semi-supervised : use all pixels, except padding
        elif supervision == "semi":
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.indices = np.array(
            [
                (x, y)
                for x, y in zip(x_pos, y_pos)
                if x >= p and x <= data.shape[0] - p and y >= p and y <= data.shape[1] - p
            ]
        )
        self.labels = [self.label[x, y] for x, y in self.indices]
        # np.random.shuffle(self.indices)

    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1 / 25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0.0, scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1 / 25):
        alpha1, alpha2 = np.random.uniform(0.01, 1.0, size=2)
        noise = np.random.normal(loc=0.0, scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert self.labels[l_indice] == value
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x, y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        # x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        # x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        # data = self.data[x1:x2, y1:y2]
        # label = self.label[x1:x2, y1:y2]
        data = self.data[(x-self.patch_size//2):(x+self.patch_size//2),(y-self.patch_size//2):(y+self.patch_size//2),:]
        # ----------------------------------------------------------------------------------------------------------------- #
        # 去掉一个标签，最后测试的时候再加上
        label = self.label[x,y]-1

        if self.flip_augmentation and self.patch_size > 1:
            # Perform data augmentation (only on 2D patches)
            data, label = self.flip(data, label)
        if self.radiation_augmentation and np.random.random() < 0.1:
            data = self.radiation_noise(data)
        if self.mixture_augmentation and np.random.random() < 0.2:
            data = self.mixture_noise(data, label)

        # data = degradation(data)
        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype="float32")
        label = np.asarray(np.copy(label), dtype="int64")

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        # Extract the center label if needed
        # if self.center_pixel and self.patch_size > 1:
        #     label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        # elif self.patch_size == 1:
        #     data = data[:, 0, 0]
        #     label = label[0, 0]

        # Add a fourth dimension for 3D CNN
        # if self.patch_size > 1:
        #     # Make 4D data ((Batch x) Planes x Channels x Width x Height)
        # data = data.unsqueeze(0)

        #----------------------------------------------------------------------------------#
        # lr_data = imresize(data,1/2)
        # bic_data = imresize(data,2)
        return data, label