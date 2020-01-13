import sys
import random
import logging
from functools import reduce
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
                file=sys.stdout,
                leave=True,
                ncols=0)


def update_pbar(pbar, loss, metric):
    postfix = {'loss': f'{loss:.3f}',
               'metric': f'{metric:.3f}'}
    pbar.set_postfix(postfix)
    pbar.update()


def get_lr(opt):
    lrs = [pg["lr"] for pg in opt.param_groups]
    res = reduce(lambda x, y: x + y, lrs) / len(lrs)
    return res


def get_logger():
    logger = logging.getLogger('back')
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s',
                                  '%d-%m-%Y %H:%M:%S')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


def set_seed(seed=None, hard=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if hard:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
