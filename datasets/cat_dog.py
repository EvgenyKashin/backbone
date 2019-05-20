from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import jpeg4py as jpeg
import torch
from torchvision import transforms
from torchvision.transforms import Normalize
from albumentations import Compose, HorizontalFlip, SmallestMaxSize, CenterCrop, RandomCrop
from backbone import utils


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
    utils.to_tensor(),
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

        return img, label

    def __len__(self):
        return len(self.data)


def get_datasets(data_dir):
    train, val = load_train_val_data(data_dir)
    return {
        'train': CatDogDataset(train, train_augs),
        'val': CatDogDataset(val, val_augs)
    }
