import torchvision as tv
import torch
from torchvision.transforms import Compose, Resize, ToTensor



def get_data(path, resolution: int, batch_size: int):
    transform = Compose([Resize((resolution, resolution)), ToTensor()])

    # 训练集
    trainset = tv.datasets.CIFAR10(
        root=path,
        train=True,
        transform=transform)

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