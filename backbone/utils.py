import sys
import numpy as np
import torch
from tqdm import tqdm


def to_tensor(scale=256):
    return lambda x: torch.from_numpy(
        x.astype(np.float32) / scale).permute(2, 0, 1)


def accuracy_metric(outputs, y_true):
    _, preds = torch.max(outputs, 1)
    return torch.sum(preds == y_true.data).double() / len(y_true)


def get_pbar(dataloader, desc):
    return tqdm(total=len(dataloader),
                desc=desc.title(),
                file=sys.stdout)