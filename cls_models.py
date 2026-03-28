# -*- coding: utf-8 -*-
# Torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.nn import init
import copy

# utils
import math
import os
import datetime
import numpy as np
import joblib

from tqdm import tqdm
from utils import grouper, sliding_window, count_sliding_window, camel_to_snake

from models import *
from dataset.dataset import degradation

def get_model(name, **kwargs):
    """
    Instantiate and obtain a model with adequate hyperparameters

    Args:
        name: string of the model name
        kwargs: hyperparameters
    Returns:
        model: PyTorch network
        optimizer: PyTorch optimizer
        criterion: PyTorch loss Function
        kwargs: hyperparameters with sane defaults
    """
    device = kwargs.setdefault("device", torch.device("cpu"))
    n_classes = kwargs["n_classes"]
    n_bands = kwargs["n_bands"]
    weights = torch.ones(n_classes)
    weights[torch.LongTensor(kwargs["ignored_labels"])] = 0.0
    weights = weights.to(device)
    weights = kwargs.setdefault("weights", weights)
    # CNN
    if name == "1DCNN":
        # 输入为4维
        kwargs.setdefault("patch_size", 16)
        center_pixel = True
        model = cnn1d(dataset=kwargs["dataset"],patch_size=kwargs['patch_size'])
        lr = kwargs.setdefault("learning_rate", 0.01)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
        kwargs.setdefault("scheduler", None) # optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=epoch // 4, verbose=True)
    elif name == "2DCNN":
        # 输入为4维
        kwargs.setdefault("patch_size", 16)
        center_pixel = True
        model = cnn2d(dataset=kwargs["dataset"])
        lr = kwargs.setdefault("learning_rate", 1e-3)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 200)
        kwargs.setdefault("batch_size", 128)
        kwargs.setdefault("scheduler", optim.lr_scheduler.MultiStepLR(optimizer, [100, 200], gamma=0.1))
    elif name == "SPRN":
        # 输入为4维
        kwargs.setdefault("patch_size", 16)
        center_pixel = True
        model = SPRN(dataset=kwargs["dataset"])
        lr = kwargs.setdefault("learning_rate", 1e-3)
        optimizer =optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 300)
        kwargs.setdefault("batch_size", 64)
        kwargs.setdefault("scheduler", optim.lr_scheduler.MultiStepLR(optimizer, [100, 250], gamma=0.1))
    elif name == "3DCNN":
        # 输入为4维
        kwargs.setdefault("patch_size", 16)
        center_pixel = True
        model = cnn3d(dataset=kwargs["dataset"],patch_size=int(kwargs['patch_size']))
        lr = kwargs.setdefault("learning_rate", 0.001)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.0005)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 300)
        kwargs.setdefault("batch_size", 100) # 3
        kwargs.setdefault("scheduler", optim.lr_scheduler.MultiStepLR(optimizer, [100, 200], gamma=0.1))
    elif name == "HybridSN":
        # 输入为4维
        kwargs.setdefault("patch_size", 16)
        center_pixel = True
        model = hybridsn(dataset=kwargs["dataset"],patch_size=int(kwargs['patch_size']))
        lr = kwargs.setdefault("learning_rate", 1e-3)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 256)
        kwargs.setdefault("scheduler", None)
    elif name == "RSSAN":
        # 输入为4维
        kwargs.setdefault("patch_size", 16)
        center_pixel = True
        model = rssan(dataset=kwargs["dataset"],patch_size=int(kwargs['patch_size']))
        lr = kwargs.setdefault("learning_rate", 0.0003)
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 200)
        kwargs.setdefault("batch_size", 16)
        kwargs.setdefault("scheduler", None)
    elif name == "M3DDCNN":
        # 输入为4维
        kwargs.setdefault("patch_size", 16)
        center_pixel = True
        model = m3ddcnn(dataset=kwargs["dataset"],patch_size=int(kwargs['patch_size'])).to(kwargs["device"])  # 这个优化器的问题导致需要先放再cuda里面
        lr = kwargs.setdefault("learning_rate", 0.01)
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 40)
        kwargs.setdefault("scheduler", None)
    elif name == "DFFN":
        # 输入为4维
        kwargs.setdefault("patch_size", 16)
        center_pixel = True
        model = dffn(dataset=kwargs["dataset"],patch_size=int(kwargs['patch_size']))
        lr = kwargs.setdefault("learning_rate", 0.1)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 200)
        kwargs.setdefault("batch_size", 100)
        kwargs.setdefault("scheduler", optim.lr_scheduler.MultiStepLR(optimizer, [100, 200], gamma=0.1))

    # Transformer
    elif name == "SSFTT":
        # 输入为4维
        kwargs.setdefault("patch_size", 16)
        center_pixel = True
        model = ssftt(dataset=kwargs["dataset"])
        lr = kwargs.setdefault("learning_rate", 1e-3)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 64)
        kwargs.setdefault("scheduler", None)
    elif name == "SpectralFormer":
        # https://github.com/danfenghong/IEEE_TGRS_SpectralFormer
        kwargs.setdefault("patch_size", 16)
        center_pixel = True
        model = spectralformer(dataset=kwargs["dataset"], patch_size=int(kwargs['patch_size']))
        lr = kwargs.setdefault("learning_rate", 5e-4)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 300) # IP:300, PU:600, Houston2013:600
        kwargs.setdefault("batch_size", 64)
        kwargs.setdefault("scheduler", optim.lr_scheduler.MultiStepLR(optimizer, [30, 60, 90, 120, 150, 180, 210, 240, 270], gamma=0.9))
    elif name == "GAHT":
        kwargs.setdefault("patch_size", 16)
        center_pixel = True
        model = gaht(dataset=kwargs["dataset"], patch_size=int(kwargs['patch_size']))
        lr = kwargs.setdefault("learning_rate", 0.001)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 300)
        kwargs.setdefault("batch_size", 64)
        kwargs.setdefault("scheduler", None)
    elif name == "MorphFormer":
        kwargs.setdefault("patch_size", 16)
        center_pixel = True
        model = MF(dataset=kwargs["dataset"], patchsize=int(kwargs['patch_size']))
        lr = kwargs.setdefault("learning_rate", 0.0005)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.005)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 500)
        kwargs.setdefault("batch_size", 64)
        kwargs.setdefault("scheduler", optim.lr_scheduler.MultiStepLR(optimizer, [50], gamma=0.9))
    elif name == "GSCViT":
        kwargs.setdefault("patch_size", 16)
        center_pixel = True
        model = gscvit(dataset=kwargs["dataset"])
        lr = kwargs.setdefault("learning_rate", 0.001)
        optimizer = optim.AdamW(model.parameters(), lr=lr,weight_decay=0.05)#optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])  # nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 200)
        kwargs.setdefault("batch_size", 128)
        kwargs.setdefault("scheduler",None)

    # CNN-Transformer
    elif name == "DCTN":
        kwargs.setdefault("patch_size", 16)
        center_pixel = True
        model = dctn(dataset=kwargs["dataset"], patchsize=int(kwargs['patch_size']))
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        epoch = kwargs.setdefault("epoch", 200)
        kwargs.setdefault(
            "scheduler",
            optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.1, patience=epoch // 4, verbose=True
            ),
        )
        kwargs.setdefault("batch_size", 100)
    elif name == "HiT":
        kwargs.setdefault("patch_size", 16)
        center_pixel = True
        model = hit(dataset=kwargs["dataset"], patchsize=int(kwargs['patch_size']))
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        epoch = kwargs.setdefault("epoch", 100)
        kwargs.setdefault(
            "scheduler",
            optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.1, patience=epoch // 4, verbose=True
            ),
        )
        kwargs.setdefault("batch_size", 32)
    elif name == "SSTMNet":
        kwargs.setdefault("patch_size", 16)
        center_pixel = True
        model = SSTMNet(dataset=kwargs["dataset"], patchsize=int(kwargs['patch_size']))
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        epoch = kwargs.setdefault("epoch", 100)
        kwargs.setdefault(
            "scheduler",
            optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.1, patience=epoch // 4, verbose=True
            ),
        )
        kwargs.setdefault("batch_size", 32)

    # multi-task
    elif name == "S2VNet":
        kwargs.setdefault("patch_size", 16)
        center_pixel = True
        model = S2VNet(band=kwargs["n_bands"], num_classes=kwargs["n_classes"], patch_size=int(kwargs['patch_size']))
        lr = kwargs.setdefault("learning_rate", 1e-3)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        epoch = kwargs.setdefault("epoch", 500)
        kwargs.setdefault("scheduler", torch.optim.lr_scheduler.StepLR(optimizer, step_size=500 // 10, gamma=0.9))
        kwargs.setdefault("batch_size", 64)
    elif name == "MCTGCL":
        kwargs.setdefault("patch_size", 16)
        center_pixel = True
        model = mctgcl(num_classes=kwargs["n_classes"], num_tokens=196)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        epoch = kwargs.setdefault("epoch", 200)
        kwargs.setdefault("scheduler", None)
        kwargs.setdefault("batch_size", 64)

    #------------------------------------------------------------------------------------------------------------------- #
    elif name[:9] == "MultiTask":
        kwargs.setdefault("patch_size", 16)
        center_pixel = True

        if name == "MultiTaskSwinCNNFuseCrossContrast":
            model = MultiTask(Swin=False, fusion=True, cross=True, contrast=True, **kwargs)

        lr = kwargs.setdefault("learning_rate", 5e-5) # 5e-4/1e-3
        optimizer = optim.AdamW(model.parameters(), lr=lr,weight_decay=1e-4)
        criterion = MultiTask_loss(cls_weight=kwargs["weights"],cls_ratio=kwargs['cls_loss_ratio'],sr_ratio=kwargs['sr_loss_ratio'])
        kwargs.setdefault("epoch", 200)
        kwargs.setdefault("batch_size", 128)
        kwargs.setdefault("scheduler", None)
    else:
        raise KeyError("{} model is unknown.".format(name))

    model = model.to(device)
    # epoch = kwargs.setdefault("epoch", 150)
    # kwargs.setdefault(
    #     "scheduler",
    #     optim.lr_scheduler.ReduceLROnPlateau(
    #         optimizer, factor=0.1, patience=epoch // 4, verbose=True
    #     ),
    # )
    # kwargs.setdefault('scheduler', None)
    kwargs.setdefault("batch_size", 64)
    kwargs.setdefault("supervision", "full")
    kwargs.setdefault("flip_augmentation", False)
    kwargs.setdefault("radiation_augmentation", False)
    kwargs.setdefault("mixture_augmentation", False)
    kwargs["center_pixel"] = center_pixel
    return model, optimizer, criterion, kwargs


def adjust_learning_rate(p, optimizer, epoch):
    """ Adjust the learning rate """

    lr = p['optimizer_kwargs']['lr']

    if p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'poly':
        lambd = pow(1 - (epoch / p['epochs']), 0.9)
        lr = lr * lambd

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

# ---------------------------------------------------------------------------------------------------------------------- #
def train(
    net,
    optimizer,
    criterion,
    data_loader,
    epoch,
    scheduler=None,
    display_iter=20,
    device=torch.device("cpu"),
    display=None,
    val_loader=None,
    supervision="full",
    name="name",
    save_dir=None
):
    """
    Training loop to optimize a network for several epochs and a specified loss

    Args:
        net: a PyTorch model
        optimizer: a PyTorch optimizer
        data_loader: a PyTorch dataset loader
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

    save_epoch = 1 #epoch // 20 if epoch > 20 else 1

    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    iter_ = 1
    loss_win, val_win = None, None
    val_accuracies = []

    for e in tqdm(range(1, epoch + 1), desc="Training the network"):
        # Set the network to training mode
        net.train()
        avg_loss = 0.0

        # Run the training loop for one epoch
        for batch_idx, (data, target) in tqdm(
            enumerate(data_loader), total=len(data_loader)
        ):
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            if supervision == "full":
                output = net(data)
                loss = criterion(output, target)
            elif supervision == "semi":
                outs = net(data)
                output, rec = outs
                loss = criterion[0](output, target) + net.aux_loss_weight * criterion[
                    1
                ](rec, data)
            else:
                raise ValueError(
                    'supervision mode "{}" is unknown.'.format(supervision)
                )
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100) : iter_ + 1])

            if display_iter and iter_ % display_iter == 0:
                string = "Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}"
                string = string.format(
                    e,
                    epoch,
                    batch_idx * len(data),
                    len(data) * len(data_loader),
                    100.0 * batch_idx / len(data_loader),
                    mean_losses[iter_],
                )
                update = None if loss_win is None else "append"
                loss_win = display.line(
                    X=np.arange(iter_ - display_iter, iter_),
                    Y=mean_losses[iter_ - display_iter : iter_],
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
            del (data, target, loss, output)

        # Update the scheduler
        avg_loss /= len(data_loader)
        if val_loader is not None:
            val_acc = val(net, val_loader, device=device, supervision=supervision)
            val_accuracies.append(val_acc)
            metric = -val_acc
        else:
            metric = avg_loss

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)
        elif scheduler is not None:
            scheduler.step()

        # Save the weights
        # if e % save_epoch == 0:
        #     save_model(
        #         net,
        #         name,
        #         data_loader.dataset.name,
        #         epoch=e,
        #         metric=abs(metric),
        #     ) # camel_to_snake(str(net.__class__.__name__))
        # -------------------------------------------------------------------------------------------------------------- #
        # 保留最好的一代
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = e
            best_model = copy.deepcopy(net)
        # -------------------------------------------------------------------------------------------------------------- #
    save_model(
        best_model,
        name,
        data_loader.dataset.name,
        epoch=best_epoch,
        metric=abs(best_val_acc),
        save_dir=save_dir
    )
    return best_epoch, best_val_acc, best_model


def test(net, img, hyperparams):
    """
    Test a model on a specific image
    """
    net.eval()
    patch_size = hyperparams["patch_size"]
    center_pixel = hyperparams["center_pixel"]
    batch_size, device = 128, hyperparams["device"] # 500
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
            output = net(data)
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
                    probs[x : x + w, y : y + h] += out
    return probs


def val(net, data_loader, device="cpu", supervision="full"):
    net.eval()
    # TODO : fix me using metrics()
    accuracy, total = 0.0, 0.0
    ignored_labels = data_loader.dataset.ignored_labels
    for batch_idx, (data, target) in enumerate(data_loader):
        with torch.no_grad():
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)
            if supervision == "full":
                output = net(data)
            elif supervision == "semi":
                outs = net(data)
                output, rec = outs
            _, output = torch.max(output, dim=1)
            for out, pred in zip(output.view(-1), target.view(-1)):
                if pred.item() in ignored_labels:
                    continue
                else:
                    accuracy += out.item() == pred.item()
                    total += 1
    net.train()
    return accuracy / total



# ------------------------------------------------------------------------------------------------------------------ #
def train_multi_contrastive(
    net,
    optimizer,
    criterion,
    data_loader,
    epoch,
    scheduler=None,
    display_iter=20,
    device=torch.device("cpu"),
    display=None,
    val_loader=None,
    supervision="full",
    name="MultiTask",
    contrastive_loss_ratio = 1e-4,
    save_dir = None
):
    """
    Training loop to optimize a network for several epochs and a specified loss

    Args:
        net: a PyTorch model
        optimizer: a PyTorch optimizer
        data_loader: a PyTorch dataset loader
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

    save_epoch = 1#epoch // 20 if epoch > 20 else 1

    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    iter_ = 1
    loss_win, val_win = None, None
    val_accuracies = []

    for e in tqdm(range(1, epoch + 1), desc="Training the network"):
        # Set the network to training mode
        net.train()
        avg_loss = 0.0

        # Run the training loop for one epoch
        for batch_idx, (data, target) in tqdm(
            enumerate(data_loader), total=len(data_loader)
        ):
            # Load the data into the GPU if required
            # lr_data, target, bic_data = lr_data.to(device), target.to(device), bic_data.to(device)
            data,target = data.to(device), target.to(device)

            optimizer.zero_grad()
            if supervision == "full":
                output_cls, output_sr, loss_contrastive = net(data)
                if loss_contrastive:
                    loss = criterion(output_cls,output_sr,target,data) + contrastive_loss_ratio * loss_contrastive #,fea_cls,fea_sr
                else:
                    loss = criterion(output_cls, output_sr, target, data)
            elif supervision == "semi":
                outs = net(data)
                output, rec = outs
                loss = criterion[0](output, target) + net.aux_loss_weight * criterion[
                    1
                ](rec, data)
            else:
                raise ValueError(
                    'supervision mode "{}" is unknown.'.format(supervision)
                )
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100) : iter_ + 1])

            if display_iter and iter_ % display_iter == 0:
                string = "Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}"
                string = string.format(
                    e,
                    epoch,
                    batch_idx * len(data),
                    len(data) * len(data_loader),
                    100.0 * batch_idx / len(data_loader),
                    mean_losses[iter_],
                )
                update = None if loss_win is None else "append"
                loss_win = display.line(
                    X=np.arange(iter_ - display_iter, iter_),
                    Y=mean_losses[iter_ - display_iter : iter_],
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
            del (data, target, loss, output_cls, output_sr)

        # Update the scheduler
        avg_loss /= len(data_loader)
        if val_loader is not None:
            val_acc = val_multi(net, val_loader, device=device, supervision=supervision)
            val_accuracies.append(val_acc)
            metric = -val_acc
        else:
            metric = avg_loss

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)
        elif scheduler is not None:
            scheduler.step()

        # Save the weights
        # if e % save_epoch == 0:
        #     save_model(
        #         net,
        #         name,
        #         data_loader.dataset.name,
        #         epoch=e,
        #         metric=abs(metric),
        #     )#camel_to_snake(str(net.__class__.__name__)),
        # -------------------------------------------------------------------------------------------------------------- #
        # 保留最好的一代
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = e
            best_model = copy.deepcopy(net)
        # -------------------------------------------------------------------------------------------------------------- #
    save_model(
        best_model,
        name,
        data_loader.dataset.name,
        epoch=best_epoch,
        metric=abs(best_val_acc),
        save_dir = save_dir
    )
    return best_epoch,best_val_acc,best_model

def test_multi(net, img, hyperparams):
    """
    Test a model on a specific image
    """
    net.eval()
    patch_size = hyperparams["patch_size"]
    center_pixel = hyperparams["center_pixel"]
    batch_size, device = hyperparams["batch_size"], hyperparams["device"] # 128
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
                # ===================================== #
                # temp_data = np.zeros((data.shape[0], data.shape[1] // 2, data.shape[2] // 2, data.shape[3]),
                #                      dtype="float32")
                # for i in range(data.shape[0]):
                #     temp_data[i] = degradation(data[i])
                # data = temp_data.transpose(0, 3, 1, 2)
                # ===================================== #
                data = data.transpose(0, 3, 1, 2)
                data = torch.from_numpy(data)
                # data = data.unsqueeze(1)

            indices = [b[1:] for b in batch]
            data = data.to(device)
            output = net.forward_cls(data)
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
                    probs[x : x + w, y : y + h] += out
    return probs


def val_multi(net, data_loader, device="cpu", supervision="full"):
    net.eval()
    # TODO : fix me using metrics()
    accuracy, total = 0.0, 0.0
    ignored_labels = data_loader.dataset.ignored_labels
    for batch_idx, (data, target) in enumerate(data_loader):
        with torch.no_grad():
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)
            if supervision == "full":
                output_cls= net.forward_cls(data)
            elif supervision == "semi":
                outs = net(data)
                output_cls, rec = outs
            _, output_cls = torch.max(output_cls, dim=1)
            for out, pred in zip(output_cls.view(-1), target.view(-1)):
                if pred.item() in ignored_labels:
                    continue
                else:
                    accuracy += out.item() == pred.item()
                    total += 1
    net.train()
    return accuracy / total

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