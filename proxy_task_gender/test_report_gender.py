from config import *
from torch.utils.data.sampler import WeightedRandomSampler
from dataset_mf import *
from model_gender import *
from config import *
from model_gender import gender_resnet

batch_size = 256

test_dataloader = DataLoader(
    test_dataset_mf, batch_size=batch_size, num_workers=NUM_WORKERS
)


ckpt = "best.ckpt"
comments = "Gender"
from model_gender import *
import engine_gender
from dataset_mf import *

model = gender_resnet(0.001)
model_name = "Resnet101_gender"
input_img_size = "224"
augment = "no"
optimizer = "SGD"
Preprocessing = "Histogram_equalization"
imbalance = "weighted_loss_upsampling"
directory = engine_gender.experiment_name


model = model.to(device)
checkpoint = torch.load(ckpt, map_location=device)
lr = checkpoint["hyper_parameters"]["lr"]
model.load_state_dict(checkpoint["state_dict"])
epochs = checkpoint["epoch"]


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
