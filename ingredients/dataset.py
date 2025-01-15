import torch
import torchvision
import random
from ingredients.utilities import set_seed
import torchvision.transforms as transforms


def get_dataset_loaders(dataset, batch_size, n_examples, seed):
    set_seed(seed=seed)

    loaders = {}

    if dataset == 'cifar10':
        print(f"Loading CIFAR10 dataset with batch size {batch_size}")
        print(f"Number of loaded images: {n_examples}")
        loaders = get_dataset_loader_cifar10(batch_size, n_examples)
    else:
        print("Please input a valid dataset (cifar10)")

    return loaders


def get_dataset_loader_cifar10(batch_size: int, n_examples: int):
    transform = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )

    dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                           download=True, transform=transform)

    subset_indices = random.sample(range(len(dataset)), n_examples)
    image_datasets_shorted = torch.utils.data.Subset(dataset, subset_indices)

    dataloaders = torch.utils.data.DataLoader(image_datasets_shorted, batch_size=batch_size,
                                                       shuffle=False, num_workers=2)

    torch.cuda.empty_cache()

    return dataloaders