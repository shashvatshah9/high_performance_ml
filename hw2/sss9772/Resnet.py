import torch.nn as nn
import torch.nn.functional as F


class ResnetBlock(nn.Module):
    expansion = 1

    def __init__(self, input_dim, output_dim, stride=1):

        super(ResnetBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            input_dim, output_dim, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(output_dim)

        self.conv2 = nn.Conv2d(output_dim, output_dim,
                               kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(output_dim)

        self.residual = nn.Sequential()

        if stride != 1 or input_dim != self.expansion*output_dim:
            self.residual = nn.Sequential(nn.Conv2d(input_dim, self.expansion*output_dim, kernel_size=1,
                                          stride=stride, bias=False), nn.BatchNorm2d(self.expansion*output_dim))

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
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_channels)

        self.layer1 = self._create_layer(
            moduleBlock, 64, num_blocks[0], stride=1)
        self.layer2 = self._create_layer(
            moduleBlock, 128, num_blocks[1], stride=2)
        self.layer3 = self._create_layer(
            moduleBlock, 256, num_blocks[2], stride=2)
        self.layer4 = self._create_layer(
            moduleBlock, 512, num_blocks[3], stride=2)

        self.linear = nn.Linear(512*moduleBlock.expansion, out_classes)

    def _create_layer(self, moduleBlock: ResnetBlock, channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for s in strides:
            layers.append(moduleBlock(self.out_channels, channels, s))
            self.out_channels = channels*moduleBlock.expansion

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


class ResnetBlockNoNorm(nn.Module):
    expansion = 1

    def __init__(self, input_dim, output_dim, stride=1):

        super(ResnetBlockNoNorm, self).__init__()

        self.conv1 = nn.Conv2d(
            input_dim, output_dim, kernel_size=3, stride=stride, padding=1, bias=False)

        self.conv2 = nn.Conv2d(output_dim, output_dim,
                               kernel_size=3, stride=1, padding=1, bias=False)

        self.residual = nn.Sequential()

        if stride != 1 or input_dim != self.expansion*output_dim:
            self.residual = nn.Sequential(nn.Conv2d(input_dim, self.expansion*output_dim, kernel_size=1,
                                          stride=stride, bias=False))

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(output)
        output = self.conv2(output)
        output += self.residual(x)
        output = F.relu(output)
        return output


class ResnetNoNorm(nn.Module):
    def __init__(self, moduleBlock: ResnetBlockNoNorm, num_blocks, out_classes=10):
        super(ResnetNoNorm, self).__init__()
        self.out_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.layer1 = self._create_layer(
            moduleBlock, 64, num_blocks[0], stride=1)
        self.layer2 = self._create_layer(
            moduleBlock, 128, num_blocks[1], stride=2)
        self.layer3 = self._create_layer(
            moduleBlock, 256, num_blocks[2], stride=2)
        self.layer4 = self._create_layer(
            moduleBlock, 512, num_blocks[3], stride=2)

        self.linear = nn.Linear(512*moduleBlock.expansion, out_classes)

    def _create_layer(self, moduleBlock: ResnetBlockNoNorm, channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for s in strides:
            layers.append(moduleBlock(self.out_channels, channels, s))
            self.out_channels = channels*moduleBlock.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
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


def ResNet18NoNorm():
    return ResnetNoNorm(ResnetBlockNoNorm, [2, 2, 2, 2])
