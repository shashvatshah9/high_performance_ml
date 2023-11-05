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

   
 
    for _, (imgs, labels) in enumerate(dataLoader):
        
        # put data on appropriate device
        imgs, labels = imgs.to(device), labels.to(device)            

        output = model(imgs)

        loss = loss_fn(output, labels)

        # backpropogation
        optim_fn.zero_grad()

       
        loss.backward()
        optim_fn.step()  
        
        curr_loss += loss.item()

        _, pred = output.max(1)
        total_labels += labels.size(0)
        correct_pred += pred.eq(labels).sum().item()
        data_loading_start = time.perf_counter()
    
    if count:
        train_loss = curr_loss/len(dataLoader)
        accuracy = 100.*(correct_pred/total_labels)
        train_losses.append(train_loss)
        train_accuracies.append(accuracy)

def train_model(trainDataLoader):
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    loss = nn.CrossEntropyLoss()
    total_start = 0
    total_end = 0
    for epoch in range(epochs):
        count = False
        if epoch == 4:
            count = True    
        training(trainDataLoader, model, loss, optimizer, epoch, count)
            


if __name__ == "__main__":
    print("main")
    epochs = 5
    compute_time_epoch = []
    data_loading_epoch = []
    comm_time_epoch = []
    cpu_gpu_epoch = []

    train_dict = { "batch_size": [], "acc" : [], "loss": [], "gpu_count": []}
   

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

    

    device_count = 4
    batch_size = 2048
       
    print(f'training for {device_count} devices and {batch_size} batch')
    model = ResNet18()
    total_params = sum(
                    param.numel() for param in model.parameters()
                )

    model = nn.DataParallel(model,device_ids = [i for i in range(device_count)])
    model.to(device)
    trainDataLoader = DataLoader(trainData, batch_size=batch_size*device_count, shuffle=True, num_workers=workers)
    train_model(trainDataLoader)

    train_dict["gpu_count"].append(device_count)
    train_dict["batch_size"].append(batch_size)
    train_dict["acc"].append(train_accuracies[-1])
    train_dict["loss"].append(train_losses[-1])
    
    pd.DataFrame.from_dict(train_dict).to_csv("output6-4.csv")

# singularity exec --nv --overlay /scratch/sss9772/rl50.ext3:rw /share/apps/images/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash
# srun --mem=8GB --gres=gpu:v100:4 --ntasks-per-node=8 --account=ece_gy_9143-2023sp --partition=n1c24m128-v100-4 --time=02:00:00 --pty /bin/bash
