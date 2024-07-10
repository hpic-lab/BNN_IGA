import os
import torch
import torchvision.datasets   as datasets
import torchvision.transforms as transforms

_DATASETS_MAIN_PATH = '/home/Datasets'
_dataset_path = {
    'cifar10':  os.path.join(_DATASETS_MAIN_PATH, 'CIFAR10'),
    'cifar100': os.path.join(_DATASETS_MAIN_PATH, 'CIFAR100'),
    'stl10':    os.path.join(_DATASETS_MAIN_PATH, 'STL10'),
    'mnist':    os.path.join(_DATASETS_MAIN_PATH, 'MNIST'),
    'imagenet': {
        'train': os.path.join(_DATASETS_MAIN_PATH, 'ImageNet/train'),
        'val':   os.path.join(_DATASETS_MAIN_PATH, 'ImageNet/val')
    }
}


def get_dataset(name, split='train', transform=None, target_transform=None, download=True):
    TrainOrVal = ((split=='train')or(split=='val'))
    if name == 'cifar10':
        if TrainOrVal:
            dataset = datasets.CIFAR10(root             = _dataset_path['cifar10'],
                                       train            = True,  
                                       transform        = transform,
                                       target_transform = target_transform,
                                       download         = download)
            val_size   = int(0.1 * len(dataset))
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            return train_dataset, val_dataset
        
        else:
            dataset = datasets.CIFAR10(root             = _dataset_path['cifar10'],
                                       train            = False,
                                       transform        = transform,
                                       target_transform = target_transform,
                                       download         = download)
            return dataset
    elif name == 'cifar100':
        if TrainOrVal:
            dataset = datasets.CIFAR10(root             = _dataset_path['cifar100'],
                                       train            = True,  
                                       transform        = transform,
                                       target_transform = target_transform,
                                       download         = download)
            val_size   = int(0.2 * len(dataset))  
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            return train_dataset, val_dataset
        
        else:
            dataset = datasets.CIFAR10(root             = _dataset_path['cifar100'],
                                       train            = False,
                                       transform        = transform,
                                       target_transform = target_transform,
                                       download         = download)
            return dataset
    elif name == 'imagenet':
        path = _dataset_path[name][split]
        return datasets.ImageFolder(root             = path,
                                    transform        = transform,
                                    target_transform = target_transform)


