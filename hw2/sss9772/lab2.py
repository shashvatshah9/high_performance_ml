import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import argparse
import time

from Resnet import ResNet18, ResNet18NoNorm

# Common training loop
def training(dataLoader, model, loss_fn, optim_fn, epoch, lenTrainDataLoader):
    print(f'training epoch: {epoch} started')
    curr_loss = 0
    correct_pred = 0
    total_labels = 0

    data_loader_time = 0
    epoch_train_time = 0

    run_start_time = time.perf_counter()
    loader_start_time = time.perf_counter()

    for _, (imgs, labels) in enumerate(dataLoader):
        # count load time
        loader_end_time = time.perf_counter()
        data_loader_time += (loader_end_time - loader_start_time)

        # put data on appropriate device
        imgs, labels = imgs.to(device), labels.to(device)

        train_start_time = time.perf_counter()
        # train images
        output = model(imgs)
        loss = loss_fn(output, labels)

        # backpropogation
        optim_fn.zero_grad()
        loss.backward()
        optim_fn.step()

        train_end_time = time.perf_counter()

        epoch_train_time += (train_end_time - train_start_time)
        curr_loss += loss.item()

        _, pred = output.max(1)
        total_labels += labels.size(0)
        correct_pred += pred.eq(labels).sum().item()
        loader_start_time = time.perf_counter()

    run_end_time = time.perf_counter()

    data_loading_time_epoch.append(data_loader_time)
    train_time_epoch.append(epoch_train_time)
    run_time_epoch.append((run_end_time - run_start_time))

    train_loss = curr_loss/lenTrainDataLoader
    accuracy = 100.*(correct_pred/total_labels)

    train_losses.append(train_loss)
    train_accuracies.append(accuracy)

    print(
        f'Epoch: {epoch} : Train Loss: {train_loss: .3f} : Accuracy : {accuracy: .3f} \n')

def train_model():
    model = ResNet18().to(device)
    optimizer = select_optimizer(args.optim, model)

    for epoch in range(epochs):
        training(trainDataLoader, model, loss, optimizer)
    total_dataloader = sum(data_loading_time)

    print("Training Done\n")
    print("DataLoader Time for Each Epoch:", data_loading_time, "\n")
    print("Training Time for Each Epoch:", train_time, "\n")
    print("Run Time for Each Epoch:", run_time)
    print("Total Data Loading time for ", num_workers,
          " worker = ", total_dataloader, "\n")


def train_model_nonorm():
    model = ResNet18NoNorm().to(device)
    optimizer = select_optimizer(args.optim, model)

    for epoch in range(epochs):
        training(trainDataLoader, model, loss, optimizer)
    total_dataloader = sum(data_loading_time)

    print("Training Done\n")
    print("DataLoader Time for Each Epoch:", data_loading_time, "\n")
    print("Training Time for Each Epoch:", train_time, "\n")
    print("Run Time for Each Epoch:", run_time)
    print("Total Data Loading time for ", num_workers,
          " worker = ", total_dataloader, "\n")

def train_with_workers(workers, device, optimarg):
    print(f'Training with {workers} worker(s) started \n\n')

    model = ResNet18().to(device)
    optimizer = select_optimizer(optimarg, model)

    loss = nn.CrossEntropyLoss()

    trainDataLoader = DataLoader(
        trainData, batch_size=128, shuffle=True, num_workers=workers)

    for epoch in range(epochs):
        training(trainDataLoader, model, loss,
                 optimizer, epoch, len(trainDataLoader))

    print(f'Training with {workers} worker(s) ended\n')

def find_optimized_workers():
    print('starting train loop')

    while (True):
        train_with_workers(workers, device, args.optim)
        loader_time_hist.append(sum(data_loading_time_epoch))
        run_time_hist.append(run_time_epoch)
        train_time_hist.append(train_time_epoch)

        print(f'Output with {workers} worker --- ')
        print(
            f'Total Dataloading time with {workers} worker(s) {loader_time_hist[-1]}')
        print(
            f'Total Train time with {workers} worker(s) for {epochs} epochs is {train_time_hist[-1]}')
        print(
            f'Total run time with {workers} worker(s) for {epochs} epochs is {run_time_hist[-1]}')

        # reinitialize
        data_loading_time_epoch = []
        train_time_epoch = []
        run_time_epoch = []

        train_accuracies = []
        train_losses = []

        if (loader_time_hist[-1] < min_time):
            min_time = loader_time_hist[-1]
        else:
            break
        workers += 4

    print(f'Optimized number of workers is {workers}')


if __name__ == "__main__":

    print('main')
    epochs = 5
    data_loading_time_epoch = []
    train_time_epoch = []
    run_time_epoch = []

    train_accuracies = []
    train_losses = []

    workers = 0
    min_time = float('inf')
    loader_time_hist = []
    train_time_hist = []
    run_time_hist = []

    train_accuracies = []
    train_losses = []

    def select_optimizer(arg, model):
        if (arg.lower() == 'sgd'):
            return optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

        if (arg.lower() == 'sgd_nest'):
            return optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)

        if (arg.lower() == 'adagrad'):
            return optim.Adagrad(model.parameters(), lr=0.01, weight_decay=5e-4)

        if (arg.lower() == 'adadelta'):
            return optim.Adadelta(model.parameters(), lr=0.1, weight_decay=5e-4)

        if (arg.lower() == 'adam'):
            return optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)

        raise Exception("Invalid optimzier argument passed")

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, required=True)
    ## not needecd for C3
    parser.add_argument('--workers', type=int, required=True)
    parser.add_argument('--optim', type=str, required=True)
    parser.add_argument('--datapath', type=str, required=True)
    args = parser.parse_args()

    device = args.device
    data_path = args.datapath

    if (device == 'gpu' and torch.cuda.is_available() == True):
        device = 'cuda'
    else:
        device = 'cpu'


    transformImage = transforms.Compose([transforms.RandomCrop((32, 32), padding=4), transforms.RandomHorizontalFlip(
        p=0.5), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    trainData = datasets.CIFAR10(
        data_path, train=True, transform=transformImage, download=True)

    
    ### FOR C2, C4, C5, C6
    ### Single chosen worker with passed in arg with chosen optimizer, runs 5 epochs
    train_model()

    ### FOR C3
    ### To find the optimized number of workers
    find_optimized_workers()

    ### FOR C7
    ### Train with Resnet with no batchnorm layer
    train_model_nonorm()