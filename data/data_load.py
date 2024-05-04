import torchvision as tv
import torch
from torchvision.transforms import Compose, Resize, ToTensor, RandomCrop, RandomHorizontalFlip, Normalize, RandomRotation
import random

augmentation_transform1 = Compose([
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

augmentation_transform2 = Compose([
    RandomRotation(10),
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])



# 定义数据增强和原数据同时存在的数据集类






def get_data(path, resolution: int, batch_size: int):
    transform = Compose(
        [
            Resize((resolution, resolution)),
            RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
            RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]
    )

    # 数据增强

    # 加载CIFAR-10数据集并应用数据增强
    train_dataset1 = tv.datasets.CIFAR10(
        root=path,
        train=True,
        transform=augmentation_transform1)

    train_dataset2 = tv.datasets.CIFAR10(
        root=path,
        train=True,
        transform=augmentation_transform2)

    # 训练集
    trainset = tv.datasets.CIFAR10(
        root=path,
        train=True,
        transform=transform)
    # trainset = torch.utils.data.ConcatDataset([trainset, train_dataset1])

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True)

    # 测试集
    testset = tv.datasets.CIFAR10(
        root=path,
        train=False,
        transform=transform)

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False)

    return trainloader, testloader