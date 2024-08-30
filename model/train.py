import torch
import torch.nn as nn
import torch.nn.utils
from torch.utils.data import DataLoader, random_split
# import wandb
import os
from model import CatClassifier
from vggnet import VGGNet
from resnet_finetune import ResNet
from preprocessing import ObeseCatDataset, transforms, target_transforms
from config import config

LOG = config.LOG
SEED = config.SEED
BATCH_SIZE = config.BATCH_SIZE
EPOCHS = config.EPOCHS
LEARNING_RATE = config.LEARNING_RATE
LOG_INTERVAL = config.LOG_INTERVAL
PRINT_INTERVAL = config.PRINT_INTERVAL
MODEL_SAVE_PATH = config.MODEL_SAVE_PATH

# model = CatClassifier()
# model = VGGNet()
model = ResNet

if LOG:
    pass
#    wandb.login()
#    wandb.init(project="obese_cats")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

data_path = 'dataset'
annotations_path = os.path.join(data_path, 'annotations.csv')
dataset = ObeseCatDataset(annotations_path, data_path, transforms) #, target_transforms)

# Train-test Split
train_set, eval_set = random_split(dataset, [0.9, 0.1], torch.Generator().manual_seed(SEED))

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)
eval_loader = DataLoader(eval_set, batch_size=BATCH_SIZE)

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print(f'Training on {device}')
print(f'NNmber of trainable parameters: {count_parameters(model):.2f}')

def train(model, dataloader, loss_fn, optimizer):
    model.train()
    for step, (images, targets) in enumerate(dataloader):
        images, targets = images.to(device), targets.to(device)

        logits = model(images)
        loss = loss_fn(logits.squeeze(), targets.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        yield step, loss.item()

@torch.no_grad()
def eval(model, dataloader, loss_fn):
    model.eval()
    losses = []
    accuracies = []
    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)

        logits = model(images)
        loss = loss_fn(logits.squeeze(), targets.squeeze())
        losses.append(loss.item())

        preds = logits.argmax(axis=1)
        labels = targets
        # labels = targets.argmax(axis=1)
        correct = (preds == labels).sum().item()
        accuracy = correct / len(labels)
        accuracies.append(accuracy) 
    return sum(losses) / len(losses), sum(accuracies) / len(accuracies)

def solver(model, train_loader, eval_loader, epochs = EPOCHS):
    print(f'Training {type(model).__name__} on {device}...')

    model.to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.95, patience=30
    )
    
    len_train_loader = len(train_loader)
    train_loss_history = []
    eval_loss_history = []

    for epoch in range(epochs):
        for step, train_loss in train(
            model, train_loader, loss_fn, optimizer
        ):
            eval_loss, eval_acc = eval(model, eval_loader, loss_fn)
            scheduler.step(eval_loss)
            # logging
            if step % LOG_INTERVAL == 0:
                eval_loss_history.append(eval_loss)
                train_loss_history.append(train_loss)
                # wandb
            # printing
            if step % PRINT_INTERVAL == 0:
                print(
                    f'Epoch: {epoch + 1:2d}/{epochs},',
                    f'Step: {step + 1:2d}/{len_train_loader},',
                    f'Validation loss: {eval_loss:.2f},',
                    f'Validation accuracy: {eval_acc:.2f}',
                    f'LR: {1e4 * scheduler.get_last_lr()[0]:.2f}e-4'
                )
    return train_loss_history, eval_loss_history

def main() -> None:
    import matplotlib.pyplot as plt
    train_losses, eval_losses = solver(
        model, train_loader, eval_loader
    )
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    plt.plot(eval_losses, label='Validation')
    plt.plot(train_losses, label='Training')
    plt.title('Training Loss Plot')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
