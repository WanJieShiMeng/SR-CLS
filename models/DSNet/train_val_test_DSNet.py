import torch
import datetime
import numpy as np
import joblib
from tqdm import tqdm
import torch.optim as optim
import copy
import os

def train_DSNet(
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
            val_acc = val_DSNet(net, val_loader, device=device, supervision=supervision)
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


def val_DSNet(net, data_loader, device="cpu", supervision="full"):
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