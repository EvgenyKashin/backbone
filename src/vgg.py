import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
import time
import os
import copy
import tqdm

import jpeg4py as jpeg
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from albumentations import Compose, HorizontalFlip, SmallestMaxSize, CenterCrop, RandomCrop
from torchvision import transforms
from torchvision.transforms import Normalize

data_dir = 'train'
input_size = 224
num_classes = 16
batch_size = 32
num_epochs = 10
num_workers = 16


def load_train_val_data(data_path):
    data = Path(data_path)

    paths = list(data.iterdir())
    labels = [int(p.name.split('.')[0] == 'dog') for p in paths]
    paths = [str(p) for p in paths]

    df = pd.DataFrame({'path': paths, 'label': labels})
    df = df.sample(frac=1.0)

    k_fold = StratifiedKFold(n_splits=10, random_state=24)
    train_ind, val_ind = next(k_fold.split(df.path, df.label))

    train_data = df.iloc[train_ind].reset_index(drop=True)
    val_data = df.iloc[val_ind].reset_index(drop=True)

    return train_data, val_data


IMAGENET_STATS = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def to_tensor(x, scale=256):
    return torch.from_numpy(
        x.astype(np.float32) / scale).permute(2, 0, 1)


resize_size = 256
img_size = [224, 224]


def get_train_augs():
    return Compose([
        SmallestMaxSize(resize_size),
        RandomCrop(*img_size),
        HorizontalFlip()
    ])


def get_val_augs():
    return Compose([
        SmallestMaxSize(resize_size),
        CenterCrop(*img_size),
    ])


train_augs = get_train_augs()
val_augs = get_val_augs()

norm_augs = transforms.Compose([
    to_tensor,
    Normalize(*IMAGENET_STATS)
])


class CatDogDataset(torch.utils.data.Dataset):
    def __init__(self, data, additional_transforms):
        self.data = data
        self.additional_transforms = additional_transforms

    def __getitem__(self, index):
        path = self.data.path[index]
        img = jpeg.JPEG(path).decode()
        label = self.data.label[index]

        img = self.additional_transforms(image=img)['image']
        img = norm_augs(img)

        return (img, label)

    def __len__(self):
        return len(self.data)


def train_model(model, dataloaders, criterion, optmizer, num_epochs):
    start_time = time.time()

    train_acc_history = []
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch: {epoch}/{num_epochs-1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm.tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optmizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optmizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'val':
                val_acc_history.append(epoch_acc)
            else:
                train_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed/60:.0f}m {time_elapsed%60:.0f}s')
    print(f'Best val acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model, train_acc_history, val_acc_history


def create_model(num_cls):
    model = models.vgg11(pretrained=False)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_cls)
    return model


train, val = load_train_val_data(data_dir)
train_dataset = CatDogDataset(train, train_augs)
val_dataset = CatDogDataset(train, val_augs)

dataloaders_dict = {
    'train': torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=num_workers),
    'val': torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                       shuffle=True, num_workers=num_workers)
}

model = create_model(num_classes)
print(model)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

criterion = nn.CrossEntropyLoss()

model, train_hist, val_hist = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs)