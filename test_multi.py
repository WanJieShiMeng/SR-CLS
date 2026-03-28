import torch
import torch.utils.data as data
import numpy as np
# Visualization
import seaborn as sns
import visdom

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
from dataset.dataset import get_dataset,DATASETS_CONFIG, HyperX
from cls_models import get_model, test_multi
import argparse
from utils import set_random_seed

def eval(dataset):
    dataset_names = [
        v["name"] if "name" in v.keys() else k for k, v in DATASETS_CONFIG.items()
    ]

    # Argument parser for CLI interaction
    parser = argparse.ArgumentParser(
        description="Run deep learning experiments on" " various hyperspectral datasets"
    )
    parser.add_argument(
        "--dataset", type=str, default=dataset, choices=dataset_names, help="Dataset to use."
    )

    parser.add_argument(
        "--restore",
        type=str,
        default="/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/IndianPines/10_label/bestModel_epoch167_0.8155.pth",
        help="Weights to use for initialization, e.g. a checkpoint",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="MultiTaskSwinCNNFuseCrossContrast",
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

    # Dataset options
    group_dataset = parser.add_argument_group("Dataset")
    group_dataset.add_argument(
        "--training_sample",
        type=float,
        default=0.1,
        help="Percentage of samples to use for training (default: 10%%) set 0.3 = get 30%",
    )
    group_dataset.add_argument(
        "--sampling_mode",
        type=str,
        help="Sampling mode" " (random sampling or disjoint, default: random, fixed, disjoint)",
        default="random",
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

    viz = visdom.Visdom(env=DATASET + " " + MODEL)
    if not viz.check_connection:
        print("Visdom is not connected. Did you run 'python -m visdom.server' ?")

    hyperparams = vars(args)
    # Load the dataset
    img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(DATASET, FOLDER, use_PCA=False)
    # ----------------------------------------------------------------------------------------------- #
    # PCA降维
    # from sklearn.decomposition import PCA
    # def applyPCA(X, numComponents):
    #     # PCA 每次结果都会不一样 因此需要设立随机种子
    #     newX = np.reshape(X, (-1, X.shape[2]))
    #     pca = PCA(n_components=numComponents, whiten=True)
    #     newX = pca.fit_transform(newX)
    #     newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    #
    #     return newX
    #
    #
    # pca_components = 30
    # img = applyPCA(img, numComponents=pca_components)
    # print('Data shape after PCA: ', img.shape)
    # ----------------------------------------------------------------------------------------------- #
    # gt标签mask边缘部分
    gt_copy = np.zeros_like(gt)
    half_patch = int(np.floor(PATCH_SIZE / 2))
    gt_copy[half_patch:-(half_patch - 1), half_patch:-(half_patch - 1)] = gt[half_patch:-(half_patch - 1),
                                                                          half_patch:-(half_patch - 1)]
    gt = gt_copy
    # ----------------------------------------------------------------------------------------------- #
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

    # ----------------------------------------------------------------------------------------------------------------- #
    seeds = [202401, 202402, 202403, 202404, 202405, 202406, 202407, 202408, 202409, 202410]

    if DATASET == 'IndianPines':
        model_path = [
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/IndianPines/10_label/bestModel_epoch115_0.8532.pth",
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/IndianPines/10_label/bestModel_epoch167_0.8155.pth",
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/IndianPines/10_label/bestModel_epoch40_0.7976.pth",
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/IndianPines/10_label/bestModel_epoch62_0.8552.pth",
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/IndianPines/10_label/bestModel_epoch157_0.8690.pth",
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/IndianPines/10_label/bestModel_epoch102_0.8472.pth",
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/IndianPines/10_label/bestModel_epoch199_0.8234.pth",
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/IndianPines/10_label/bestModel_epoch72_0.8452.pth",
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/IndianPines/10_label/bestModel_epoch174_0.8393.pth",
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/IndianPines/10_label/bestModel_epoch65_0.8313.pth"
        ]
    elif DATASET == 'Salinas':
        model_path = [
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/Salinas/10_label/bestModel_epoch81_0.9577.pth",
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/Salinas/10_label/bestModel_epoch81_0.9374.pth",
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/Salinas/10_label/bestModel_epoch47_0.9463.pth",
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/Salinas/10_label/bestModel_epoch64_0.9681.pth",
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/Salinas/10_label/bestModel_epoch78_0.9489.pth",
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/Salinas/10_label/bestModel_epoch140_0.9577.pth",
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/Salinas/10_label/bestModel_epoch126_0.9151.pth",
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/Salinas/10_label/bestModel_epoch129_0.9466.pth",
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/Salinas/10_label/bestModel_epoch137_0.9652.pth",
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/Salinas/10_label/bestModel_epoch96_0.9514.pth"
        ]
    elif DATASET == 'KSC':
        model_path = [
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/KSC/10_label/bestModel_epoch189_0.9094.pth",
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/KSC/10_label/bestModel_epoch182_0.8937.pth",
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/KSC/10_label/bestModel_epoch121_0.8780.pth",
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/KSC/10_label/bestModel_epoch166_0.8504.pth",
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/KSC/10_label/bestModel_epoch178_0.8701.pth",
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/KSC/10_label/bestModel_epoch103_0.8780.pth",
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/KSC/10_label/bestModel_epoch133_0.8858.pth",
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/KSC/10_label/bestModel_epoch151_0.9094.pth",
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/KSC/10_label/bestModel_epoch135_0.9094.pth",
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/KSC/10_label/bestModel_epoch184_0.8583.pth"
        ]
    elif DATASET == 'WHUHC':
        model_path = [
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/WHUHC/10_label/bestModel_epoch176_0.8977.pth",
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/WHUHC/10_label/bestModel_epoch99_0.8996.pth",
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/WHUHC/10_label/bestModel_epoch135_0.8805.pth",
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/WHUHC/10_label/bestModel_epoch115_0.8764.pth",
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/WHUHC/10_label/bestModel_epoch136_0.8624.pth",
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/WHUHC/10_label/bestModel_epoch160_0.8765.pth",
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/WHUHC/10_label/bestModel_epoch163_0.8530.pth",
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/WHUHC/10_label/bestModel_epoch127_0.8468.pth",
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/WHUHC/10_label/bestModel_epoch110_0.9025.pth",
            "/home/wxy/multi_task_sr/checkpoints/MultiTaskSwinCNNFuseCrossContrast/WHUHC/10_label/bestModel_epoch162_0.8688.pth"
        ]
    results = []

    for run in range(N_RUNS):
        set_random_seed(seeds[run])
        # data
        train_gt, test_gt = sample_gt(gt, SAMPLE_PERCENTAGE, mode=SAMPLING_MODE)
        print(
            "{} samples selected (over {})".format(
                np.count_nonzero(train_gt), np.count_nonzero(gt)
            )
        )
        display_predictions(convert_to_color(train_gt), viz, caption="Train ground truth")
        display_predictions(convert_to_color(test_gt), viz, caption="Test ground truth")

        # Neural network
        model, _, _, hyperparams = get_model(MODEL, **hyperparams)
        print(hyperparams)

        if CHECKPOINT is not None:
            print("加载模型")
            model.load_state_dict(torch.load(model_path[run]))

        # val_dataset = HyperX(img, test_gt, **hyperparams)
        # val_loader = data.DataLoader(
        #     val_dataset,
        #     # pin_memory=hyperparams['device'],
        #     batch_size=hyperparams["batch_size"],
        # )
        # val_acc = val_multi(model, val_loader, device=hyperparams['device'], supervision=hyperparams['supervision'])
        # print(val_acc)

        probabilities = test_multi(model, img, hyperparams)
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
        import os
        save_dir = "./checkpoints/ours_final_result" + "/" + DATASET + "/" + str(SAMPLE_PERCENTAGE) + "_label_p=" + str(PATCH_SIZE) + "_b=" + str(hyperparams['batch_size']) +"/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        path = os.path.join(save_dir, str(SAMPLE_PERCENTAGE) + "_label_avg_results.txt")
        show_results(results, viz, label_values=LABEL_VALUES, agregated=True)
if __name__ == '__main__':
    eval('IndianPines')
    eval('Salinas')
    eval('KSC')
    eval('WHUHC')