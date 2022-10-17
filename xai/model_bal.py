from config import *
import torch.nn.functional as F
from dataset_p import *


class balance_resnet(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet101(pretrained=True)
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.model.fc = nn.Linear(in_features=2048, out_features=2, bias=True)

    def forward(self, x):
        out = self.model(x)
        return out
