import torch
import torch.nn as nn
import torch.optim as optim
from backbone import Bone, utils
from datasets import cifar10
from models.resnet import resnet20

data_dir = 'cifar10'
model_name = 'resnet'
num_classes = 10
batch_size = 256
epochs_count = 200
num_workers = 12

datasets = cifar10.get_datasets(data_dir)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if model_name == 'resnet':
    model = resnet20(num_classes=num_classes)

model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                      weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

bone = Bone(model,
            datasets,
            criterion,
            optimizer,
            metric_fn=utils.accuracy_metric,
            metric_increase=True,
            batch_size=batch_size,
            num_workers=num_workers,
            weights_path=f'weights/best_{model_name}.pth',
            log_dir=f'logs/{model_name}')

bone.fit(epochs_count)
