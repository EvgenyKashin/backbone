import torch
import torch.nn as nn
import torch.optim as optim
from backbone import Bone, utils
from datasets import cifar10
from models.resnet import resnet20
from models.custom_resnet import se_resnet20, srm_resnet20

data_dir = 'cifar10'
model_name = 'srmnet'
num_classes = 10
batch_size = 256
epochs_count = 150
num_workers = 12

datasets = cifar10.get_datasets(data_dir)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if model_name == 'resnet':
    model = resnet20(num_classes=num_classes)  # 0.877 - 272,474
elif model_name == 'senet':
    model = se_resnet20(num_classes=num_classes)  # 0.877 - 274,490(+0.7%)
elif model_name == 'srmnet':
    model = srm_resnet20(num_classes=num_classes)  # and 0.871 - 273,818( +0.5%)

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
