import time

import matplotlib.pyplot as plt

import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

import models

ds_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True)
])

ds_train = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ds_transform
)
ds_test = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ds_transform
)

batch_size = 64
dataloader_train = torch.utils.data.DataLoader(
    ds_train,
    batch_size=batch_size,
    shuffle=True
)
dataloader_test = torch.utils.data.DataLoader(
    ds_test,
    batch_size=batch_size
)

for image_batch, label_batch in dataloader_test:
    print(image_batch.shape)
    print(label_batch.shape)
    break

model = models.MyModel()
acc_test = models.test_accuracy(model, dataloader_test)
print(f'test_accuracy: {acc_test*100:.3f}%')

model = models.MyModel()

loss_fn = torch.nn.CrossEntropyLoss()

learning_rate = 1e-3
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

acc_test = models.test_accuracy(model, dataloader_test)
print(f'test accuracy: {acc_test*100:.2f}%')

models.train(model, dataloader_test, loss_fn, optimizer)

acc_test = models.test_accuracy(model, dataloader_test)
print(f'test accuracy: {acc_test*100:.2f}%')
