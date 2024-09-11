import os
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
from preprocessing import ObeseCatDataset, transforms
from torchvision.models import resnet18

data_dir = "dataset"
test_dir = os.path.join(data_dir, "test")
annotations_test = os.path.join(test_dir, "annotations_test.csv")

test_set = ObeseCatDataset(annotations_test, test_dir)


def _prepare_image(img: torch.Tensor, transforms, n: int = 32) -> torch.Tensor:
    """
    applies random transforms n times to produce a batch of images,
    used to obtain better predictions
    """
    return torch.stack([transforms(img) for _ in range(n)])


def predict_img(
    model: nn.Module,
    img: torch.Tensor,
    transforms: Callable[[torch.Tensor], torch.Tensor] = transforms,
    n: int = 32,
) -> int:
    """
    Generates predictions for one image by first applying random transforms
    n times to create a batch, then making predictions on the batch, and
    finally averaging the predictions.
    """
    batch: torch.Tensor = _prepare_image(img, transforms, n)
    logits: torch.Tensor = F.softmax(model(batch), dim=1)
    logit: torch.Tensor = torch.mean(logits, dim=0)
    pred: int = int(torch.argmax(logit).item())
    return pred


def load_resnet18(model_path: str) -> nn.Module:
    model = resnet18(weights=None)
    num_classes = 3
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


@torch.no_grad()
def main() -> None:
    MODEL_PATH = config.MODEL_SAVE_PATH
    model = load_resnet18(MODEL_PATH)
    model.cpu()

    preds = []
    labels = []
    length = len(test_set)

    for idx, (img, label) in enumerate(test_set):
        pred = predict_img(model, img)
        preds.append(pred)
        labels.append(label)
        print(f"{idx:3d}/{length} - Prediction: {pred}, Label: {label}")

    preds = torch.tensor(preds)
    labels = torch.tensor(labels)
    correct = sum(preds == labels)
    accuracy = correct / len(labels)
    print(f"Test accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()
