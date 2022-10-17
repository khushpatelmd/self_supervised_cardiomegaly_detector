from numpy import reciprocal
from config import *


class autoencoder(LightningModule):
    def __init__(self, hidden_dim=1024, pred_num=2, lr=0.05):
        super().__init__()
        self.save_hyperparameters()
        self.hidden_dim = hidden_dim
        self.pred_num = pred_num
        self.first_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.first_encoder = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.second_encoder = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.third_encoder = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.encoding_mlp = torch.nn.Linear(128 * 20 * 20, self.hidden_dim)
        self.decoding_mlp = torch.nn.Linear(self.hidden_dim, 128 * 20 * 20)
        self.classifier_block = torch.nn.Sequential(
            nn.ReLU(),
            torch.nn.Linear(int(self.hidden_dim), int(self.hidden_dim / 2)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(self.hidden_dim / 2), int(self.hidden_dim / 4)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(self.hidden_dim / 4), self.pred_num),
        )

        self.first_decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
        )

        self.second_decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
        )

        self.third_decoder = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
        )

        self.last_cnn = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.valid_loss = torch.nn.CrossEntropyLoss()
        self.test_loss = torch.nn.CrossEntropyLoss()
        self.train_loss = torch.nn.CrossEntropyLoss()

        self.MSE_loss = torch.nn.MSELoss(
            size_average=None, reduce=None, reduction="none"
        )

        self.testauc = torchmetrics.AUROC(num_classes=None)
        self.testf1 = torchmetrics.F1(average="micro", num_classes=2)

        self.valauc = torchmetrics.AUROC(num_classes=None)
        self.valf1 = torchmetrics.F1(num_classes=2, average="micro")

        self.auc = torchmetrics.AUROC(num_classes=None)
        self.f1 = torchmetrics.F1(num_classes=2, average="micro")

    def forward(self, x):
        # encoder
        x = self.first_cnn(x)
        x = self.first_encoder(x)
        x = self.second_encoder(x)
        x = self.third_encoder(x)
        shape = x.size()
        enc_features = torch.flatten(x, start_dim=1, end_dim=-1)
        lin1 = self.encoding_mlp(enc_features)
        dec = self.decoding_mlp(lin1)
        dec = dec.view(shape)
        dec = self.first_decoder(dec)
        dec = self.second_decoder(dec)  # 32, 40, 40, 40
        dec = self.third_decoder(dec)  # 16, 80, 80, 80
        recon = self.last_cnn(dec)  # 1, 80, 80, 80
        logits = self.classifier_block(lin1)
        return logits, recon

    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        logits, recon = self(x)
        loss = self.train_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        prob = F.softmax(logits, dim=-1)[:, -1]
        recon_loss = self.MSE_loss(recon, x)
        recon_loss = recon_loss * mask
        recon_loss = torch.sum(recon_loss)
        recon_loss = recon_loss / mask.sum()
        total_loss = recon_loss + loss
        train_auc = self.auc(prob, y)
        train_f1 = self.f1(preds, y)
        self.log("train_auc_step", train_auc, on_step=False, on_epoch=True)
        self.log("train_f1", train_f1, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        self.log("recon_train_loss", recon_loss)
        self.log("total_loss", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        logits, recon = self(x)
        loss = self.valid_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        recon_loss = self.MSE_loss(recon, x)
        recon_loss = recon_loss * mask
        recon_loss = recon_loss.sum()
        recon_loss = recon_loss / mask.sum()
        total_loss = recon_loss + loss
        val_f1 = self.valf1(preds, y)
        self.log(f"val_loss", loss, prog_bar=True)
        self.log(f"val_f1", val_f1, on_step=False, on_epoch=True, prog_bar=True)

        return {"val_loss": loss, "val_f1": val_f1}

    def test_step(self, batch, batch_idx):
        x, y, mask = batch
        logits, _ = self(x)
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

        lr_scheduler_config = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                "max",
                patience=3,
                min_lr=self.hparams["lr"] / 1000,
                factor=0.5,
            ),
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_f1",
            "strict": True,
            "name": None,
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }


if __name__ == "__main__":
    model2 = autoencoder(512, 2)

    img = torch.randn(size=(16, 1, 160, 160))

    logits, recon = model2(img)
    print("Khush")
