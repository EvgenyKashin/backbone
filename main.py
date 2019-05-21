import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from backbone import Bone, utils
from models.vgg import vgg11
from datasets import cat_dog

data_dir = 'train'
input_size = 224
num_classes = 2
batch_size = 32
epochs_count = 20
num_workers = 8

datasets = cat_dog.get_datasets(data_dir)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = vgg11(num_classes, batch_norm=True)

model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
criterion = nn.CrossEntropyLoss()

bone = Bone(model,
            datasets,
            criterion,
            optimizer,
            scheduler,
            scheduler_after_ep=False,
            metric_fn=utils.accuracy_metric,
            metric_increase=True,
            batch_size=batch_size,
            num_workers=num_workers)

bone.fit(epochs_count)
