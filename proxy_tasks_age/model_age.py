from config import *
import torch.nn.functional as F
from dataset_age import *


def create_model_age():
    model = torchvision.models.resnet101(pretrained=True)
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    model.fc = nn.Linear(in_features=2048, out_features=1, bias=True)
    return model


# def create_model():
#     model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=2)
#     return model


class age_resnet(LightningModule):
    def __init__(self, lr=0.05):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_model_age()
        self.val_loss = torch.nn.MSELoss()
        self.test_loss = torch.nn.MSELoss()
        self.train_loss = torch.nn.MSELoss()

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = (F.sigmoid(self(x))).squeeze()
        loss = self.train_loss(preds, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = (F.sigmoid(self(x))).squeeze()
        loss = self.val_loss(preds, y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = (F.sigmoid(self(x))).squeeze()
        loss = self.test_loss(preds, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )

        # optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        # steps_per_epoch = len(train_dataloader)
        # scheduler_dict = {
        #     "scheduler": OneCycleLR(
        #         optimizer,
        #         max_lr=0.001,
        #         epochs=self.trainer.max_epochs,
        #         steps_per_epoch=steps_per_epoch,
        #     ),
        #     "interval": "step",
        # }
        # return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
        lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": ReduceLROnPlateau(
                optimizer,
                "min",
                patience=3,
                min_lr=self.hparams["lr"] / 1000,
                factor=0.5,
            ),
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "val_loss",
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            "strict": True,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": None,
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }
        # return optimizer

