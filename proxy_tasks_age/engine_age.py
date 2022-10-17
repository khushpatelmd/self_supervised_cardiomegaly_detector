experiment_name = "Resnet_age"
lr = 1e-2
img_size = 224
batch_size = 164
# custom imports
from torch.utils.data.sampler import WeightedRandomSampler
from dataset_age import *
from model_age import age_resnet
from config import *

train_dataloader = DataLoader(
    train_dataset_age, batch_size=batch_size, num_workers=NUM_WORKERS, shuffle=True
)
val_dataloader = DataLoader(
    valid_dataset_age, batch_size=batch_size, num_workers=NUM_WORKERS
)
test_dataloader = DataLoader(
    test_dataset_age, batch_size=batch_size, num_workers=NUM_WORKERS
)

# parameters
NUM_WORKERS = int(os.cpu_count() / 2)


model = age_resnet(lr)

dir_name = experiment_name

lr_monitor = LearningRateMonitor(logging_interval="epoch")
model_checkpoint = ModelCheckpoint(
    dirpath=dir_name,
    monitor="val_loss",
    save_last=True,
    filename="{epoch}-{val_loss:.2f}",
    save_top_k=5,
)
tb_logger = TensorBoardLogger(save_dir=dir_name + "/tb_logs")
csv_logger = CSVLogger(save_dir=dir_name + "/csv_logs")

es = EarlyStopping(patience=5, monitor="val_loss", mode="min")

if __name__ == "__main__":
    trainer = Trainer(
        num_sanity_val_steps=2,
        # resume_from_checkpoint="/home/kpatel38/c_drive/School_project/bmi6331-data-challenge/Resnet101_1/last.ckpt",
        logger=[tb_logger, csv_logger],
        gpus=[0],
        callbacks=[lr_monitor, model_checkpoint],
        # track_grad_norm=2,
        # weights_summary="full",
        sync_batchnorm=True,
        log_every_n_steps=20,
        fast_dev_run=False,
        benchmark=True,
        max_epochs=15,
    )

    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
    trainer.test(model, test_dataloaders=test_dataloader)
