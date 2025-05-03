from typing import Literal, Tuple
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar10_dataset(train: bool) -> datasets.CIFAR10:
    return datasets.CIFAR10(
        root=".",
        train=train,
        download=True,
        transform=transforms.Compose(
            [
                *(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, padding=4),
                    ]
                    if train
                    else []
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
    )


def get_mnist_dataset(train: bool) -> datasets.MNIST:
    return datasets.MNIST(
        root=".",
        train=train,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                transforms.Lambda(lambda x: x.flatten()),
            ]
        ),
    )


def get_dataloaders(
    dataset_id: Literal["cifar10", "mnist"], batch_size: int = 128
) -> Tuple[DataLoader, DataLoader, int]:
    if dataset_id == "cifar10":
        train_loader = DataLoader(
            get_cifar10_dataset(train=True), batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            get_cifar10_dataset(train=False), batch_size=batch_size, shuffle=False
        )
        num_classes = 10
    elif dataset_id == "mnist":
        train_loader = DataLoader(
            get_mnist_dataset(train=True), batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            get_mnist_dataset(train=False), batch_size=batch_size, shuffle=False
        )
        num_classes = 10
    else:
        raise ValueError(f"Invalid dataset ID: {dataset_id}")

    return train_loader, test_loader, num_classes
