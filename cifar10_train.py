import torch
import torch.nn as nn
import torch.optim as optim
from backbone import Bone, utils
from datasets import cifar10
from models.vgg import vgg11

data_dir = 'cifar10'
num_classes = 10
batch_size = 32
epochs_count = 20
num_workers = 8

datasets = cifar10.get_datasets(data_dir)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = vgg11(num_classes, batch_norm=True)

model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                      weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

bone = Bone(model,
            datasets,
            criterion,
            optimizer,
            metric_fn=utils.accuracy_metric,
            metric_increase=True,
            batch_size=batch_size,
            num_workers=num_workers)

bone.fit(epochs_count)
