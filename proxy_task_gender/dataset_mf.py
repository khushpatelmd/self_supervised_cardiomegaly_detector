import torch
import skimage
import pandas as pd

class xrayds_mf(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        transforms1,
        transforms2,
        as_float=True,
        seperate_transform=False,
        monai_transforms=True,
    ):
        super().__init__()
        self.dataset = pd.read_csv(dataset)
        self.images = list(self.dataset["FileName"])
        self.Cardiomegaly = list(self.dataset["PatGender"])

        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.as_float = as_float
        self.seperate_transform = seperate_transform
        self.monai_transforms = monai_transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idxx=int):

        if self.as_float == True:
            img = skimage.img_as_float(
                skimage.io.imread("../images/" + str(self.images[idxx]), as_gray=True)
            )
        elif self.as_float == False:
            img = skimage.img_as_ubyte(
                skimage.io.imread("../images/" + str(self.images[idxx]), as_gray=True)
            )
        label = torch.tensor(self.Cardiomegaly[idxx], dtype=torch.long)

        if self.monai_transforms == False:
            if self.seperate_transform == False:
                augs = self.transforms1(image=img)
                img = augs["image"]
            if self.seperate_transform == True:
                if label == 0:
                    augs = self.transforms1(image=img)
                    img = augs["image"]
                elif label == 1:
                    augs = self.transforms2(image=img)
                    img = augs["image"]

        elif self.monai_transforms == True:
            img = self.transforms1(img)

        return img, label
