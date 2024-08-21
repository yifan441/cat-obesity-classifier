from train import model, train_loader, eval_loader, solver
from config import config
from preprocessing import ObeseCatDataset, transforms, target_transforms
import torch
from helpers import plot
import os
import matplotlib.pyplot as plt

# solver(model, train_loader, eval_loader)

data_dir = 'dataset'
test_dir = os.path.join(data_dir, 'test')
annotations_test = os.path.join(test_dir, 'annotations_test.csv')

test_set = ObeseCatDataset(annotations_test, data_dir, transforms) #, target_transforms)

@torch.no_grad()
def main() -> None:
    model.cpu()
    for img, label in test_set:
        pred = model(img.unsqueeze(0))
        print(pred)
        print(label)
        print(pred.squeeze())

if __name__ == "__main__":
    main()
