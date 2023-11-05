import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import pandas as pd
import time


class ResnetBlock(nn.Module):
    expansion = 1

    def __init__(self, input_dim, output_dim, stride=1):
        super(ResnetBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            input_dim, output_dim, kernel_size=3, stride=stride, padding=1, bias=False
        )

        self.bn1 = nn.BatchNorm2d(output_dim)

        self.conv2 = nn.Conv2d(
            output_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.bn2 = nn.BatchNorm2d(output_dim)

        self.residual = nn.Sequential()

        if stride != 1 or input_dim != self.expansion * output_dim:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    input_dim,
                    self.expansion * output_dim,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * output_dim),
            )

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = F.relu(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output += self.residual(x)
        output = F.relu(output)
        return output


class Resnet(nn.Module):
    def __init__(self, moduleBlock: ResnetBlock, num_blocks, out_classes=10):
        super(Resnet, self).__init__()
        self.out_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_channels)

        self.layer1 = self._create_layer(moduleBlock, 64, num_blocks[0], stride=1)
        self.layer2 = self._create_layer(moduleBlock, 128, num_blocks[1], stride=2)
        self.layer3 = self._create_layer(moduleBlock, 256, num_blocks[2], stride=2)
        self.layer4 = self._create_layer(moduleBlock, 512, num_blocks[3], stride=2)

        self.linear = nn.Linear(512 * moduleBlock.expansion, out_classes)

    def _create_layer(self, moduleBlock: ResnetBlock, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for s in strides:
            layers.append(moduleBlock(self.out_channels, channels, s))
            self.out_channels = channels * moduleBlock.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = F.relu(output)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)

        output = F.avg_pool2d(output, 4)
        output = output.view(output.size(0), -1)
        output = self.linear(output)
        return output


def ResNet18():
    return Resnet(ResnetBlock, [2, 2, 2, 2])


# Common training loop
def training(dataLoader, model, loss_fn, optim_fn, epoch, count):
    print(f"training epoch: {epoch} started")
    curr_loss = 0
    correct_pred = 0
    total_labels = 0

    epoch_train_time = 0

    for _, (imgs, labels) in enumerate(dataLoader):
        # count load time
        if count:
            train_start_time = time.perf_counter()

        # put data on appropriate device
        imgs, labels = imgs.to(device), labels.to(device)

        # train images
        output = model(imgs)
        loss = loss_fn(output, labels)

        # backpropogation
        optim_fn.zero_grad()
        loss.backward()
        optim_fn.step()
        if count:
            train_end_time = time.perf_counter()
            epoch_train_time += train_end_time - train_start_time

        curr_loss += loss.item()

        _, pred = output.max(1)
        total_labels += labels.size(0)
        correct_pred += pred.eq(labels).sum().item()

    if count:
        train_time_epoch.append(epoch_train_time)


def train_model(trainDataLoader):
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    loss = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        count = False
        if epoch == 1:
            count = True
        training(trainDataLoader, model, loss, optimizer, epoch, count)

    print("Training Time for Each Epoch:", train_time_epoch[-1], "\n")
    # print("Run Time for Each Epoch:", run_time)


if __name__ == "__main__":
    print("main")
    epochs = 2
    train_time_epoch = []
    train_dict = {"time": [], "batch_size": []}
    train_accuracies = []
    train_losses = []

    workers = 2
    min_time = float("inf")
    train_time_hist = []

    train_accuracies = []
    train_losses = []

    device = "cuda"

    transformImage = transforms.Compose(
        [
            transforms.RandomCrop((32, 32), padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainData = datasets.CIFAR10(
        './', train=True, transform=transformImage, download=True
    )

    gpus_connected = torch.cuda.device_count()

    # experiment 1
    batch_size = 32
    model = ResNet18().to(device)
    while True:
        trainDataLoader = DataLoader(
            trainData, batch_size=batch_size, shuffle=True, num_workers=workers
        )
        try:
            print("Starting training with batch size", batch_size, "\n")
            train_model(trainDataLoader)
            train_dict["batch_size"].append(batch_size)
            train_dict["time"].append(train_time_epoch[-1])
            batch_size *= 4
        except RuntimeError as e:
            print(
                f"GPU memory full at batch size : {batch_size}, use {batch_size/4} for 1 GPU"
            )
            break
        

    pd.DataFrame.from_dict(train_dict).to_csv("output6-1.csv")