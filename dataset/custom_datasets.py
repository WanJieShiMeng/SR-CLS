from utils import open_file

CUSTOM_DATASETS_CONFIG = {
    "DFC2018_HSI": {
        "img": "2018_IEEE_GRSS_DFC_HSI_TR.HDR",
        "gt": "2018_IEEE_GRSS_DFC_GT_TR.tif",
        "download": False,
        "loader": lambda folder: dfc2018_loader(folder),
    },
    "WHUHC":{
        "img": "WHU_Hi_HanChuan.mat",
        "gt": "WHU_Hi_HanChuan_gt.mat",
        "download": False,
        "loader": lambda folder: WHU_Hi_HanChuan(folder),
    }
}


def WHU_Hi_HanChuan(folder):
    img = open_file(folder + "WHU_Hi_HanChuan.mat")["WHU_Hi_HanChuan"]
    gt = open_file(folder + "WHU_Hi_HanChuan_gt.mat")["WHU_Hi_HanChuan_gt"]
    gt = gt.astype("uint8")

    rgb_bands = (29, 18, 17) # [48,29,17], [67,48,17]

    label_values = [
        "Unclassified",
        "Strawberry",
        "Cowpea",
        "Soybean",
        "Sorghum",
        "Water spinach",
        "Watermelon",
        "Greens",
        "Trees",
        "Grass",
        "Red roof",
        "Gray roof",
        "Plastic",
        "Bare soil",
        "Road",
        "Bright object",
        "Water",
    ]
    ignored_labels = [0]
    palette = None
    return img, gt, rgb_bands, ignored_labels, label_values, palette

def dfc2018_loader(folder):
    img = open_file(folder + "2018_IEEE_GRSS_DFC_HSI_TR.HDR")[:, :, :-2]
    gt = open_file(folder + "2018_IEEE_GRSS_DFC_GT_TR.tif")
    gt = gt.astype("uint8")

    rgb_bands = (47, 31, 15)

    label_values = [
        "Unclassified",
        "Healthy grass",
        "Stressed grass",
        "Artificial turf",
        "Evergreen trees",
        "Deciduous trees",
        "Bare earth",
        "Water",
        "Residential buildings",
        "Non-residential buildings",
        "Roads",
        "Sidewalks",
        "Crosswalks",
        "Major thoroughfares",
        "Highways",
        "Railways",
        "Paved parking lots",
        "Unpaved parking lots",
        "Cars",
        "Trains",
        "Stadium seats",
    ]
    ignored_labels = [0]
    palette = None
    return img, gt, rgb_bands, ignored_labels, label_values, palette