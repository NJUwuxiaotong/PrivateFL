import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from constant import consts


def get_dataset(dataset_name):
    dataset_dir = consts.DATASET_ROOT_DIR
    if dataset_name.upper() == consts.DATASET_MNIST:
        train_set, valid_set = _build_mnist(dataset_dir, normalize=True)
    else:
        print("Error: No dataset named %s!" % dataset_name)
        exit(1)
    return train_set, valid_set


def construct_data_loaders(dataset, batch_size=0, shuffle=True,
                           drop_last=False):
    if consts.MULTITHREAD_DATAPROCESSING:
        num_workers = min(
            torch.get_num_threads(),
            consts.MULTITHREAD_DATAPROCESSING) if torch.get_num_threads() > 1 \
            else 0
    else:
        num_workers = 0

    train_loader = DataLoader(
        dataset, batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle, drop_last=drop_last, num_workers=num_workers,
        pin_memory=consts.PIN_MEMORY)
    return train_loader, DataStats(dataset)


def get_labels_from_loader(dataset_loader, batch_index=0):
    batch_id = 0
    dataset_iter = iter(dataset_loader)
    for examples, labels in dataset_iter:
        if batch_id == batch_index:
            return labels
        batch_id = batch_id + 1

    print("Error: Batch index - %s exceeds the range - %s" %
          (batch_index, len(dataset_iter)))
    exit(1)


class DataStats(object):
    def __init__(self, dataset):
        self.example_no = len(dataset)
        if isinstance(dataset, Subset):
            self.class_no = len(dataset.dataset.classes)
        else:
            self.class_no = len(dataset.classes)
        # tensor: (access_num, row, column)
        self.example_shape = dataset[0][0].shape


def _build_mnist(data_path, augmentations=True, normalize=True):
    """Define MNIST with everything considered."""
    data_mean = consts.MNIST_MEAN[0]
    data_std = consts.MNIST_STD[0]

    # Load data
    train_set = torchvision.datasets.MNIST(
        root=data_path, train=True, download=True,
        transform=transforms.ToTensor())
    valid_set = torchvision.datasets.MNIST(
        root=data_path, train=False, download=True,
        transform=transforms.ToTensor())

    """
    if mnist_mean is None:
        cc = torch.cat(
            [train_set[i][0].reshape(-1) for i in range(len(train_set))], dim=0)
        data_mean = (torch.mean(cc, dim=0).item(),)
        data_std = (torch.std(cc, dim=0).item(),)
    else:
        data_mean, data_std = mnist_mean, mnist_std
    """

    # Organize pre_processing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std)
        if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        train_set.transform = transform_train
    else:
        train_set.transform = transform
    valid_set.transform = transform
    return train_set, valid_set
