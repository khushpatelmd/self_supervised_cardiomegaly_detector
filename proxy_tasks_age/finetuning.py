from config import *

ckpt = "/home/kpatel38/c_drive/School_project/src/proxy_tasks_age/best_age.ckpt"
checkpoint = torch.load(ckpt, map_location=torch.device("cpu"))
from dataset_p import *


model = torchvision.models.resnet101(pretrained=True)
model.conv1 = nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
model.fc = nn.Linear(in_features=2048, out_features=1, bias=True)


class age_resnet(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet101(pretrained=True)
        self.model.conv1 = nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(in_features=2048, out_features=1, bias=True)        

    def forward(self, x):
        out = self.model(x)
        return out
      
model = age_resnet()

model.load_state_dict(checkpoint["state_dict"])

#freezing layers
for mod in model.children():
    child_counter = 0
    for child in mod.children():
        if child_counter==0:
            for param in child.parameters():
                param.requires_grad=False
        if child_counter<=6:
                for c in child.children():
                    for name, param in c.named_parameters():
                        param.requires_grad=False
        child_counter += 1

model.model.bn1.weight.requires_grad = False
model.model.bn1.bias.requires_grad = False
num_final_in = model.model.fc.in_features
model.model.fc = nn.Linear(num_final_in, 2)
optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=0.001,
            momentum=0.9)

experiment_name = "Resnet_age_finetuning"
img_size = 224
batch_size = 128


train_dataloader = DataLoader(
    train_dataset_bal, batch_size=batch_size, num_workers=NUM_WORKERS, shuffle=True
)
val_dataloader = DataLoader(
    valid_dataset_bal, batch_size=batch_size, num_workers=NUM_WORKERS
)
test_dataloader = DataLoader(
    test_dataset_bal, batch_size=batch_size, num_workers=NUM_WORKERS
)

class engine_ft(LightningModule):
    def __init__(self, lr=0.05):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.valid_loss = torch.nn.CrossEntropyLoss()
        self.test_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.valid_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.log("val_loss", loss, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.test_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.log(f"test_loss", loss, prog_bar=True)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
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
            "monitor": "val_loss",
            "strict": True,
            "name": None,
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        } 
        
model_ft = engine_ft(0.001)

dir_name = experiment_name

model_checkpoint = ModelCheckpoint(
    dirpath=dir_name,
    monitor="val_loss",
    save_last=True,
    filename="{epoch}-{val_loss:.2f}",
    save_top_k=5,
)
tb_logger = TensorBoardLogger(save_dir=dir_name + "/tb_logs")
csv_logger = CSVLogger(save_dir=dir_name + "/csv_logs")

if __name__ == "__main__":
    trainer = Trainer(
        num_sanity_val_steps=2,
        logger=[tb_logger, csv_logger],
        gpus=[0],
        callbacks=[model_checkpoint],
        log_every_n_steps=4,
        fast_dev_run=False,
        benchmark=True,
        max_epochs=500,
    )

    trainer.fit(
        model_ft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
    
from config import *

# ckpt = "/home/kpatel38/c_drive/School_project/bmi6331-data-challenge/src/Resnet_age_finetuning/last.ckpt"
comments = "Resnet age ft"

model_name = "Resnet101_age"

model = model.to(device)
# checkpoint = torch.load(ckpt, map_location=device)
# lr = checkpoint["hyper_parameters"]["lr"]
# model.load_state_dict(checkpoint["state_dict"])
# epochs = checkpoint["epoch"]

test_dataloader = DataLoader(test_dataset_bal, batch_size=16, num_workers=12)

model.eval()
y_true = []
y_pred = []
y_prob = []
with torch.no_grad():
    for test_data in test_dataloader:
        test_images, test_labels = (
            test_data[0].to(device),
            test_data[1].to(device),
        )

        logits = F.softmax(model(test_images), dim=1)
        preds = torch.argmax(logits, dim=1)
        prob = logits[:, 1]
        y_true.extend(test_labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(prob.cpu().numpy())

report = classification_report(
    y_true, y_pred, target_names=["none", "cardiomyopathy"], output_dict=True
)

metrics = [
    {
#         "model_name": model_name,
#         "Preprocessing": Preprocessing,
#         "imbalance": imbalance,
#         "input_img_size": input_img_size,
#         "augment": augment,
#         "optimizer": optimizer,
#         "lr": lr,
#         "epochs": epochs,
        "auc": roc_auc_score(y_true, y_prob),
        "Sensitivity": report["cardiomyopathy"]["recall"],
        "Specificity": report["none"]["recall"],
        "F1 score": f1_score(y_true, y_pred, average="micro"),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Balanced accuracy": balanced_accuracy_score(y_true, y_pred),
#         "comments": comments,
    }
]

fieldnames = [
    "model_name",
    "Preprocessing",
    "imbalance",
    "input_img_size",
    "augment",
    "optimizer",
    "lr",
    "epochs",
    "auc",
    "Sensitivity",
    "Specificity",
    "F1 score",
    "MCC",
    "Balanced accuracy",
    "comments",
    "directory",
]

with open("experiments.csv", "a") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(metrics)

print("Completed successfully writing", metrics)
