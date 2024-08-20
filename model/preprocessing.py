import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Lambda, v2


class ObeseCatDataset(Dataset):
    def __init__(
        self, annotations_file, data_dir, transform=None, target_transform=None
    ):
        self.labels = pd.read_csv(annotations_file)
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.key = {0: "skinny", 1: "normal", 2: "obese"}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.labels.iloc[idx, 0]
        image = read_image(img_path)
        label = self.labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def visualize(self, idx):
        img_path = self.labels.iloc[idx, 0]
        image = read_image(img_path)
        label = self.labels.iloc[idx, 1]
        plt.imshow(image.permute(1, 2, 0))
        plt.title(f"ID: {idx}, Class: {self.key[label]}")
        plt.show()

transforms = v2.Compose([
    v2.Resize(size=(300, 300)),
    v2.RandomResizedCrop(
        size=(224, 224),
        scale = (0.7, 1.0),
        ratio = (0.95, 1.05),
        antialias=True
    ),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=30),
    # v2.RandomPerspective(distortion_scale=0.3, p=0.2),
    v2.ToDtype(torch.float32, scale=True),
])

def one_hot(y):
    return torch.zeros(3, dtype=torch.float).scatter_(
        dim=0, index=torch.tensor(y), value=1
    )


def main() -> None:
    from helpers import plot
    dataset_path = "dataset"
    annotations_path = os.path.join(dataset_path, "annotations.csv")
    dataset = ObeseCatDataset(annotations_path, dataset_path)
    print(f"Length of dataset: {len(dataset)}")
    response = input("Would you like to see some cats? (y/n)")
    while response != "n":
        idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[idx]
        row1 = [[img] + [transforms(img) for _ in range(3)]]
        rows2_4 = [[transforms(img) for _ in range(4)] for _ in range(3)]
        plot(row1 + rows2_4)
        plt.suptitle(f'ID: {idx}, Class: {dataset.key[label]}')
        plt.show()
        print("Would you like to see more cats? (y/n)")
        response = input()
    print('Goodbye.')

    # print("Dataset with transforms:")
    # transformed = ObeseCatDataset(
    #     annotations_path,
    #     dataset_path,
    #     transforms,
    #     Lambda(one_hot)
    # )
    # print(f"Check an idx [{0}-{len(transformed)}]:")
    # idx = int(input())
    # img, label = transformed[idx]
    # transformed.visualize(idx)
    # print(img.shape)
    # print(label)
    # from helpers import plot
    # print("Check an idx:")
    # idx = int(input())
    # img, label = dataset[idx]
    # # resize_cropper = v2.RandomResizedCrop(size=(224, 224))
    # # resized_crops = [resize_cropper(img) for _ in range(4)]
    # transformed = [transforms(img) for _ in range(4)]
    # # plot([img] + resized_crops)
    # plot([img] + transformed)
    # plt.show()


if __name__ == "__main__":
    main()
