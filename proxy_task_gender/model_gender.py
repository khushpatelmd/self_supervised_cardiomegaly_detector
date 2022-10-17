from config import *
import torch.nn.functional as F
from dataset_p import *


def create_model():
    model = torchvision.models.resnet101(pretrained=True)
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    model.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
    return model


# def create_model():
#     model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=2)
#     return model


class gender_resnet(LightningModule):
    def __init__(self, lr=0.05):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_model()
        self.valid_loss = torch.nn.CrossEntropyLoss()
        self.test_loss = torch.nn.CrossEntropyLoss()
        self.train_loss = torch.nn.CrossEntropyLoss()
        self.testauc = torchmetrics.AUROC(num_classes=None)
        self.testf1 = torchmetrics.F1(average="micro", num_classes=2)

        self.valauc = torchmetrics.AUROC(num_classes=None)
        self.valf1 = torchmetrics.F1(num_classes=2, average="micro")

        self.auc = torchmetrics.AUROC(num_classes=None)
        self.f1 = torchmetrics.F1(num_classes=2, average="micro")

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        # weights = torch.FloatTensor([1, 22])
        # weights = weights.type_as(x)
        # loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        # loss = loss_fn(logits, y)
        loss = self.train_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        prob = F.softmax(logits, dim=-1)[:, -1]
        train_auc = self.auc(prob, y)
        train_f1 = self.f1(preds, y)
        self.log("train_auc_step", train_auc, on_step=False, on_epoch=True)
        self.log("train_f1", train_f1, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.valid_loss(logits, y)
        preds = torch.argmax(logits, dim=1)

        val_f1 = self.valf1(preds, y)
        self.log(f"val_loss", loss, prog_bar=True)
        self.log(f"val_f1", val_f1, on_step=False, on_epoch=True, prog_bar=True)

        return {"val_loss": loss, "val_f1": val_f1}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.test_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        prob = F.softmax(logits, dim=-1)[:, -1]

        test_auc = self.testauc(prob, y)
        test_f1 = self.testf1(preds, y)

        self.log(f"test_loss", loss, prog_bar=True)
        self.log(f"test_f1", test_f1, on_step=False, on_epoch=True)
        self.log("test_auc", test_auc, on_step=False, on_epoch=True)
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
                "max",
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
            "monitor": "val_f1",
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
