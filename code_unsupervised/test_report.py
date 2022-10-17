from config import *
from torch.utils.data.sampler import WeightedRandomSampler
from model import *
from config import *

batch_size = 256

ckpt = "ae_ss_weighted/best.ckpt"
comments = "SSAE_weighted_loss"
from model import *
import engine

model = autoencoder()
model_name = "AESS"
input_img_size = "160"
augment = "no"
optimizer = "SGD"
Preprocessing = "Histogram_equalization"
imbalance = "weighted_loss_upsampling"
directory = engine.experiment_name


model = model.to(device)
checkpoint = torch.load(ckpt, map_location=device)
lr = checkpoint["hyper_parameters"]["lr"]
model.load_state_dict(checkpoint["state_dict"])
epochs = checkpoint["epoch"]
test_dataloader = DataLoader(test_dataset_bal, batch_size=1, num_workers=NUM_WORKERS)


model.eval()
y_true = []
y_pred = []
y_prob = []
imgs = []
recons = []
with torch.no_grad():
    for test_data in test_dataloader:
        test_images, test_labels = (
            test_data[0].to(device),
            test_data[1].to(device),
        )
        logits, recon = model(test_images)
        logits = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        prob = logits[:, 1]
        y_true.extend(test_labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(prob.cpu().numpy())
        recon = np.squeeze(recon.cpu().numpy())
        recons.append(recon)
        imgs.append(np.squeeze(test_images.cpu().numpy()))

report = classification_report(
    y_true, y_pred, target_names=["none", "cardiomyopathy"], output_dict=True
)

metrics = [
    {
        "model_name": model_name,
        "Preprocessing": Preprocessing,
        "imbalance": imbalance,
        "input_img_size": input_img_size,
        "augment": augment,
        "optimizer": optimizer,
        "lr": lr,
        "epochs": epochs,
        "auc": roc_auc_score(y_true, y_prob),
        "Sensitivity": report["cardiomyopathy"]["recall"],
        "Specificity": report["none"]["recall"],
        "F1 score": f1_score(y_true, y_pred, average="micro"),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Balanced accuracy": balanced_accuracy_score(y_true, y_pred),
        "comments": comments,
        "directory": directory,
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
    writer.writerows(metrics)

print("Completed successfully writing", metrics)

# counter = 0
# for i in recons[:10]:
#     plt.imsave(
#         f"/home/kpatel38/c_drive/School_project/src/code_self_supervised/reconstructed images/with_mask_2/test_recon{counter}.jpg",
#         i,
#         cmap="gray",
#     )
#     counter += 1

# counter = 0
# for i in imgs[:10]:
#     plt.imsave(
#         f"/home/kpatel38/c_drive/School_project/src/code_self_supervised/reconstructed images/with_mask_2/test_image{counter}.jpg",
#         i,
#         cmap="gray",
#     )
#     counter += 1
