import torch
import datetime
import numpy as np
import joblib
from tqdm import tqdm
import torch.optim as optim
import copy
import os

def train_S2VNet(
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
        for batch_idx, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            # 
            re_unmix_nonlinear, re_unmix, batch_pred, edm_var_1, edm_var_2, feature_abu, edm_per = net(data)

            band = re_unmix.shape[1] // 2  # 2 represents the number of decoder layer
            output_linear = re_unmix[:, 0:band] + re_unmix[:, band:band * 2]
            re_unmix = re_unmix_nonlinear + output_linear

            # compute kl loss
            kl_div = -0.5 * (edm_var_2 + 1 - edm_var_1 ** 2 - edm_var_2.exp())
            kl_div = kl_div.sum() / batch_pred.shape[0]
            kl_div = torch.max(kl_div, torch.tensor(0).cuda())

            # compute tv loss
            edm_per_diff = edm_per[1:, :] - edm_per[:(edm_per.shape[0] - 1), :]
            edm_per_diff = edm_per_diff.abs()
            loss_tv = edm_per_diff.mean()  # endmember tv_loss

            b_x, h_x, w_x = feature_abu.shape[0], feature_abu.shape[-2], feature_abu.shape[-1]
            h_tv = torch.pow((feature_abu[:, :, 1:, :] - feature_abu[:, :, :h_x - 1, :]), 2).sum()
            w_tv = torch.pow((feature_abu[:, :, :, 1:] - feature_abu[:, :, :, :w_x - 1]), 2).sum()
            loss_tv_abu = (h_tv + w_tv) / (b_x * 2 * h_x * w_x)  # abundance tv_loss

            sad_loss = torch.mean(torch.acos(torch.sum(data * re_unmix, dim=1) /
                                             (torch.norm(re_unmix, dim=1, p=2) * torch.norm(data, dim=1,
                                                                                            p=2) + 1e-5)))
            loss = criterion(batch_pred,
                             target) + sad_loss + 0.01 * kl_div + 0.01 * loss_tv + 0.01 * loss_tv_abu
            
            # 
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
            del (data, target, loss)

        # Update the scheduler
        avg_loss /= len(data_loader)
        if val_loader is not None:
            val_acc = val_S2VNet(net, val_loader, device=device)
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
    save_model(
        best_model,
        name,
        data_loader.dataset.name,
        epoch=best_epoch,
        metric=abs(best_val_acc),
        save_dir=save_dir
    )
    return best_epoch, best_val_acc, best_model


def val_S2VNet(net, data_loader, device="cpu"):
    net.eval()
    # TODO : fix me using metrics()
    accuracy, total = 0.0, 0.0
    ignored_labels = data_loader.dataset.ignored_labels
    for batch_idx, (data, target) in enumerate(data_loader):
        with torch.no_grad():
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)

            re_unmix_nonlinear, re_unmix, output, edm_var_1, edm_var_2, _, _ = net(data)

            band = re_unmix.shape[1] // 2  # 2 represents the number of decoder layer
            output_linear = re_unmix[:, 0:band] + re_unmix[:, band:band * 2]
            re_unmix = re_unmix_nonlinear + output_linear

            sad_loss = torch.mean(torch.acos(torch.sum(data * re_unmix, dim=1) /
                                             (torch.norm(re_unmix, dim=1, p=2) * torch.norm(data, dim=1, p=2))))
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
def test_S2VNet(net, img, hyperparams):
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
            re_unmix_nonlinear, re_unmix, output, edm_var_1, edm_var_2, _, _ = net(data)
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