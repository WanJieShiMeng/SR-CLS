# -*- coding: utf-8 -*-
"""
DEEP LEARNING FOR HYPERSPECTRAL DATA.

This script allows the user to run several deep models (and SVM baselines)
against various hyperspectral datasets. It is designed to quickly benchmark
state-of-the-art CNNs on various public hyperspectral datasets.

This code is released under the GPLv3 license for non-commercial and research
purposes only.
For commercial use, please contact the authors.
"""
# Python 2/3 compatiblity
from __future__ import print_function
from __future__ import division

import copy

# Torch
import torch
import torch.utils.data as data
from torchsummary import summary

# Numpy, scipy, scikit-image, spectral
import numpy as np
import sklearn.svm
import sklearn.model_selection
from skimage import io

# Visualization
import seaborn as sns
import visdom

import os
from utils import (
    metrics,
    convert_to_color_,
    convert_from_color_,
    display_dataset,
    display_predictions,
    explore_spectrums,
    plot_spectrums,
    sample_gt,
    build_dataset,
    show_results,
    compute_imf_weights,
    get_device,
)
from dataset.dataset import get_dataset, HyperX_multi, open_file, DATASETS_CONFIG, HyperX
from cls_models import get_model, train, test, save_model, test_multi, train_multi_contrastive

import argparse

def train_each(dataset_name, model_name, training_sample):
    dataset_names = [
        v["name"] if "name" in v.keys() else k for k, v in DATASETS_CONFIG.items()
    ]

    # Argument parser for CLI interaction
    parser = argparse.ArgumentParser(
        description="Run deep learning experiments on" " various hyperspectral datasets"
    )
    parser.add_argument(
        "--dataset", type=str, default=dataset_name, choices=dataset_names, help="Dataset to use."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=model_name,
        help="Model to train. Available:\n"
        "SVM (linear), "
        "SVM_grid (grid search on linear, poly and RBF kernels), "
        "baseline (fully connected NN), "
        "hu (1D CNN), "
        "hamida (3D CNN + 1D classifier), "
        "lee (3D FCN), "
        "chen (3D CNN), "
        "li (3D CNN), "
        "he (3D CNN), "
        "luo (3D CNN), "
        "sharma (2D CNN), "
        "boulch (1D semi-supervised CNN), "
        "liu (3D semi-supervised CNN), "
        "mou (1D RNN)",
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="Folder where to store the "
        "datasets (defaults to the current working directory).",
        default="./Datasets/",
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=1,
        help="Specify CUDA device (defaults to -1, which learns on CPU)",
    )
    parser.add_argument("--runs", type=int, default=10, help="Number of runs (default: 1)")
    parser.add_argument(
        "--restore",
        type=str,
        default=None,
        help="Weights to use for initialization, e.g. a checkpoint",
    )

    # Dataset options
    group_dataset = parser.add_argument_group("Dataset")
    group_dataset.add_argument(
        "--training_sample",
        type=float,
        default=training_sample,
        help="Percentage of samples to use for training (default: 10%%) set 0.3 = get 30%",
    )
    group_dataset.add_argument(
        "--sampling_mode",
        type=str,
        help="Sampling mode" " (random sampling or disjoint, default: random, fixed, disjoint)",
        default="fixed",
    )
    group_dataset.add_argument(
        "--train_set",
        type=str,
        default=None,
        help="Path to the train ground truth (optional, this "
        "supersedes the --sampling_mode option)",
    )
    group_dataset.add_argument(
        "--test_set",
        type=str,
        default=None,
        help="Path to the test set (optional, by default "
        "the test_set is the entire ground truth minus the training)",
    )
    # Training options
    group_train = parser.add_argument_group("Training")
    group_train.add_argument(
        "--epoch",
        type=int,
        help="Training epochs (optional, if" " absent will be set by the model)",
    )
    group_train.add_argument(
        "--patch_size",
        type=int,
        help="Size of the spatial neighbourhood (optional, if "
        "absent will be set by the model)",
        default=16
    )
    group_train.add_argument(
        "--lr", type=float, help="Learning rate, set by the model if not specified."
    )
    group_train.add_argument(
        "--class_balancing",
        action="store_true",
        help="Inverse median frequency class balancing (default = False)",
    )
    group_train.add_argument(
        "--batch_size",
        type=int,
        help="Batch size (optional, if absent will be set by the model",
    )
    group_train.add_argument(
        "--test_stride",
        type=int,
        default=1,
        help="Sliding window step stride during inference (default = 1)",
    )
    # Data augmentation parameters
    group_da = parser.add_argument_group("Data augmentation")
    group_da.add_argument(
        "--flip_augmentation", action="store_true", help="Random flips (if patch_size > 1)"
    )
    group_da.add_argument(
        "--radiation_augmentation",
        action="store_true",
        help="Random radiation noise (illumination)",
    )
    group_da.add_argument(
        "--mixture_augmentation", action="store_true", help="Random mixes between spectra"
    )

    parser.add_argument(
        "--with_exploration", action="store_true", help="See data exploration visualization",
    )
    parser.add_argument(
        "--download",
        type=str,
        default=None,
        nargs="+",
        choices=dataset_names,
        help="Download the specified datasets and quits.",
    )
    # --------------------------------------------------------------------------- #
    # loss 权重设置
    parser.add_argument(
        "--cls_loss_ratio",
        type=float,
        default=1,
        nargs="+",
        choices=dataset_names,
        help="Download the specified datasets and quits.",
    )
    parser.add_argument(
        "--contrastive_loss_ratio",
        type=float,
        default=1e-4,
        nargs="+",
        choices=dataset_names,
        help="Download the specified datasets and quits.",
    )
    parser.add_argument(
        "--sr_loss_ratio",
        type=float,
        default=0.1,
        nargs="+",
        choices=dataset_names,
        help="Download the specified datasets and quits.",
    )


    args = parser.parse_args()

    CUDA_DEVICE = get_device(args.cuda)

    # % of training samples
    SAMPLE_PERCENTAGE = args.training_sample
    # Data augmentation ?
    FLIP_AUGMENTATION = args.flip_augmentation
    RADIATION_AUGMENTATION = args.radiation_augmentation
    MIXTURE_AUGMENTATION = args.mixture_augmentation
    # Dataset name
    DATASET = args.dataset
    # Model name
    MODEL = args.model
    # Number of runs (for cross-validation)
    N_RUNS = args.runs
    # Spatial context size (number of neighbours in each spatial direction)
    PATCH_SIZE = args.patch_size
    # Add some visualization of the spectra ?
    DATAVIZ = args.with_exploration
    # Target folder to store/download/load the datasets
    FOLDER = args.folder
    # Number of epochs to run
    EPOCH = args.epoch
    # Sampling mode, e.g random sampling
    SAMPLING_MODE = args.sampling_mode
    # Pre-computed weights to restore
    CHECKPOINT = args.restore
    # Learning rate for the SGD
    LEARNING_RATE = args.lr
    # Automated class balancing
    CLASS_BALANCING = args.class_balancing
    # Training ground truth file
    TRAIN_GT = args.train_set
    # Testing ground truth file
    TEST_GT = args.test_set
    TEST_STRIDE = args.test_stride

    if args.download is not None and len(args.download) > 0:
        for dataset in args.download:
            get_dataset(dataset, target_folder=FOLDER)
        quit()

    viz = visdom.Visdom(env=DATASET + " " + MODEL)
    if not viz.check_connection:
        print("Visdom is not connected. Did you run 'python -m visdom.server' ?")


    hyperparams = vars(args)
    # ----------------------------------------------------------------------------------------------- #
    if DATASET in ["IndianPines", "WHUHC"]:
        hyperparams['batch_size'] = 128
    else:
        hyperparams['batch_size'] = 64
    # ----------------------------------------------------------------------------------------------- #

    # Load the dataset
    img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(DATASET, FOLDER, use_PCA=False)
    img = np.pad(img,((PATCH_SIZE // 2, PATCH_SIZE // 2 - 1),(PATCH_SIZE // 2, PATCH_SIZE // 2 - 1),(0,0)), 'symmetric')
    gt = np.pad(gt,((PATCH_SIZE // 2, PATCH_SIZE // 2 - 1),(PATCH_SIZE // 2, PATCH_SIZE // 2 - 1)), 'constant',constant_values=0)
    print(img.shape,gt.shape)

    # Number of classes
    N_CLASSES = len(LABEL_VALUES)
    # Number of bands (last dimension of the image tensor)
    N_BANDS = img.shape[-1]

    if palette is None:
        # Generate color palette
        palette = {0: (0, 0, 0)}
        for k, color in enumerate(sns.color_palette("hls", len(LABEL_VALUES) - 1)):
            palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype="uint8"))
    invert_palette = {v: k for k, v in palette.items()}


    def convert_to_color(x):
        return convert_to_color_(x, palette=palette)


    def convert_from_color(x):
        return convert_from_color_(x, palette=invert_palette)


    # Instantiate the experiment based on predefined networks
    hyperparams.update(
        {
            "n_classes": N_CLASSES,
            "n_bands": N_BANDS,
            "ignored_labels": IGNORED_LABELS,
            "device": CUDA_DEVICE,
            # "center_pixel":True,
            # "supervision":"full"
        }
    )
    hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

    # Show the image and the ground truth
    display_dataset(img, gt, RGB_BANDS, LABEL_VALUES, palette, viz)
    color_gt = convert_to_color(gt)

    if DATAVIZ:
        # Data exploration : compute and show the mean spectrums
        mean_spectrums = explore_spectrums(
            img, gt, LABEL_VALUES, viz, ignored_labels=IGNORED_LABELS
        )
        plot_spectrums(mean_spectrums, viz, title="Mean spectrum/class")

    # ------------------------------------------------------------------------------------------------------------------ #
    # 固定种子
    from utils import set_random_seed
    seeds = [202401, 202402, 202403, 202404, 202405, 202406, 202407, 202408, 202409, 202410]
    # ------------------------------------------------------------------------------------------------------------------ #

    # 设置模型和结果的保存路径
    save_dir = "./checkpoints/" + MODEL + "/" + DATASET + "/" + str(SAMPLE_PERCENTAGE) + "_label_p=" + str(PATCH_SIZE) + "_b=" + str(hyperparams['batch_size']) +"/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    results = []
    # run the experiment several times
    for run in range(N_RUNS):
        set_random_seed(seeds[run])

        if TRAIN_GT is not None and TEST_GT is not None:
            train_gt = open_file(TRAIN_GT)
            test_gt = open_file(TEST_GT)
        elif TRAIN_GT is not None:
            train_gt = open_file(TRAIN_GT)
            test_gt = np.copy(gt)
            w, h = test_gt.shape
            test_gt[(train_gt > 0)[:w, :h]] = 0
        elif TEST_GT is not None:
            test_gt = open_file(TEST_GT)
        else:
            # Sample random training spectra
            train_gt, test_gt = sample_gt(gt, SAMPLE_PERCENTAGE, mode=SAMPLING_MODE)

            # if CLASS_BALANCING:
            #     weights = compute_imf_weights(trainval_gt, N_CLASSES, IGNORED_LABELS)
            #     hyperparams["weights"] = torch.from_numpy(weights)
            # Neural network
        model, optimizer, loss, hyperparams = get_model(MODEL, **hyperparams)

        # Split train set in train/val
        if dataset_name == "WHUHC":
            val_num = 0.01
        else:
            val_num = 0.05
        val_gt, test_gt = sample_gt(test_gt, val_num, mode="random")
        print("------------------------DATA SPLIT------------------------")
        print(
            "Train: {} samples selected (over {})".format(
                np.count_nonzero(train_gt), np.count_nonzero(gt)
            )
        )
        print(
            "Val: {} samples selected (over {})".format(
                np.count_nonzero(val_gt), np.count_nonzero(gt)
            )
        )
        print(
            "Test: {} samples selected (over {})".format(
                np.count_nonzero(test_gt), np.count_nonzero(gt)
            )
        )
        print("------------------------Validation: Number of samples per class------------------------")
        for i in range(N_CLASSES):
            if i==0:
                continue
            else:
                print("{}:{}".format(LABEL_VALUES[i], np.sum(val_gt==i)))
        print("------------------------Test: Number of samples per class------------------------")
        for i in range(N_CLASSES):
            if i==0:
                continue
            else:
                print("{}:{}".format(LABEL_VALUES[i], np.sum(test_gt==i)))
        print("------------------------Train------------------------")
        print(
            "Running an experiment with the {} model".format(MODEL),
            "run {}/{}".format(run + 1, N_RUNS),
        )

        # display_predictions(convert_to_color(trainval_gt), viz, caption="TrainVal ground truth")
        display_predictions(convert_to_color(train_gt), viz, caption="Train ground truth")
        display_predictions(convert_to_color(val_gt), viz, caption="Val ground truth")
        display_predictions(convert_to_color(test_gt), viz, caption="Test ground truth")

        print(hyperparams)
        train_dataset = HyperX(data=img, gt=train_gt, is_multi_task=False, **hyperparams)
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=hyperparams["batch_size"],
            pin_memory=True,
            shuffle=True,
        )
        val_dataset = HyperX(data=img, gt=val_gt, is_multi_task=False, **hyperparams)
        val_loader = data.DataLoader(
            val_dataset,
            pin_memory=True,
            batch_size=hyperparams["batch_size"], # hyperparams["batch_size"]
        )

        if CHECKPOINT is not None:
            model.load_state_dict(torch.load(CHECKPOINT))

        try:
            if MODEL[:9]=="MultiTask":
                best_epoch, best_val_acc, best_model = train_multi_contrastive(
                    model,
                    optimizer,
                    loss,
                    train_loader,
                    hyperparams["epoch"],
                    scheduler=hyperparams["scheduler"],
                    device=hyperparams["device"],
                    supervision=hyperparams["supervision"],
                    val_loader=val_loader,  # val_loader
                    display=viz,
                    name=MODEL,
                    contrastive_loss_ratio=hyperparams['contrastive_loss_ratio'],
                    save_dir=save_dir
                )
            else:
                best_epoch, best_val_acc, best_model = train(
                    model,
                    optimizer,
                    loss,
                    train_loader,
                    hyperparams["epoch"],
                    scheduler=hyperparams["scheduler"],
                    device=hyperparams["device"],
                    supervision=hyperparams["supervision"],
                    val_loader=val_loader,  # val_loader
                    display=viz,
                    name=MODEL,
                    save_dir=save_dir
                )
            text = "Dataset:{}, Model:{}--Training Results:  best_epoch:{}, best_valacc:{}".format(dataset_name, model_name, best_epoch, best_val_acc)
            viz.text(text)

        except KeyboardInterrupt:
            # Allow the user to stop the training
            pass
        if MODEL[:9] == "MultiTask":
            probabilities = test_multi(best_model, img, hyperparams)
        else:
            probabilities = test(best_model, img, hyperparams)
        # --------------------------------------------------------------------- #
        prediction = np.argmax(probabilities, axis=-1)

        run_results = metrics(
            prediction,
            test_gt,
            ignored_labels=hyperparams["ignored_labels"],
            n_classes=N_CLASSES,
        )

        mask = np.zeros(gt.shape, dtype="bool")
        for l in IGNORED_LABELS:
            mask[gt == l] = True
        prediction[mask] = 0

        color_prediction = convert_to_color(prediction)
        display_predictions(
            color_prediction,
            viz,
            gt=convert_to_color(test_gt),
            caption="Prediction vs. test ground truth",
        )

        results.append(run_results)
        show_results(run_results, viz, label_values=LABEL_VALUES)

    if N_RUNS > 1:
        path =  os.path.join(save_dir,"avg_results.txt")
        show_results(results, viz, label_values=LABEL_VALUES, agregated=True, path=path)

if __name__ == "__main__":
    dataset_names = ["IndianPines","Salinas", "KSC", "WHUHC"]#["IndianPines","Salinas", "KSC", "WHUHC", "PaviaU", "KSC","PaviaC"]
    train_sample = [10,10,10,10]#[0.05 * 2, 0.01 * 2, 0.005 * 2]
    model_names = ["MultiTaskSwinCNNFuseCrossContrast"]

    for i in range(len(model_names)):
        for j in range(len(dataset_names)):
            train_each(dataset_names[j],model_names[i],train_sample[j])