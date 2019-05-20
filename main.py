import torch
import torch.nn as nn
import torch.optim as optim
from backbone import Bone, utils
from models.vgg import vgg11
from datasets import cat_dog

data_dir = 'train'
input_size = 224
num_classes = 2
batch_size = 32
num_epochs = 20
num_workers = 24

datasets = cat_dog.get_datasets(data_dir)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = vgg11(num_classes, batch_norm=True)
print(model)

model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

bone = Bone(model,
            datasets,
            criterion,
            optimizer,
            utils.accuracy_metric,
            batch_size,
            num_workers)

bone.train(num_epochs)