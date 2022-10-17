img_size = 224

# torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torch
from torch.utils.data import WeightedRandomSampler, DataLoader

device = torch.device("cuda:0")
import torchmetrics
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR

# pl
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ProgressBar,
    EarlyStopping,
)
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger


# monai
from monai.networks.nets import EfficientNetBN
from monai.transforms import (
    Activations,
    AddChannel,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    EnsureType,
    Resize,
    Lambda,
    ToTensor,
)

# python
import csv
import os
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import pandas as pd
import skimage
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pickle

transforms = A.Compose(
    [A.Equalize(), A.Resize(img_size, img_size), A.ToFloat(), ToTensorV2()]
)
transforms_monai_train = Compose(
    [
        AddChannel(),
        Resize(
            (128, 128)
        ),  # set this arbitrarily, need to experiment with different sizes
        ScaleIntensity(),
        RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
        # RandFlip(spatial_axis=0, prob=0.5), # might not be needed, try without quickly
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),  # test if this makes it better
        ToTensor(),
    ]
)

transforms_monai_test = Compose(
    [
        AddChannel(),
        Resize(
            (128, 128)
        ),  # set this arbitrarily, need to experiment with different sizes
        ScaleIntensity(),
        EnsureType(),
        ToTensor(),
    ]
)

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    classification_report,
)

transforms_A = A.Compose([A.Equalize(), A.Resize(224, 224), A.ToFloat(), ToTensorV2()])

# Custom imports
from dataset_age import *
from dataset_p import *

# paths
train_data = "../data/train.csv"
val_data = "../data/val.csv"
test_data = "../data/test.csv"

# configs
NUM_WORKERS = int(os.cpu_count() / 2)

test_dataset_age = xrayds_age(
    test_data,
    transforms_A,
    None,
    as_float=False,
    seperate_transform=False,
    monai_transforms=False,
)
train_dataset_age = xrayds_age(
    train_data,
    transforms_A,
    None,
    as_float=False,
    seperate_transform=False,
    monai_transforms=False,
)


valid_dataset_age = xrayds_age(
    val_data,
    transforms_A,
    None,
    as_float=False,
    seperate_transform=False,
    monai_transforms=False,
)


train_dataset_bal = xrayds_p(
    "../data/bal_train.csv",
    transforms_A,
    None,
    as_float=False,
    seperate_transform=False,
    monai_transforms=False,
)
valid_dataset_bal = xrayds_p(
    "../data/bal_val.csv",
    transforms_A,
    None,
    as_float=False,
    seperate_transform=False,
    monai_transforms=False,
)
test_dataset_bal = xrayds_p(
    test_data,
    transforms_A,
    None,
    as_float=False,
    seperate_transform=False,
    monai_transforms=False,
)