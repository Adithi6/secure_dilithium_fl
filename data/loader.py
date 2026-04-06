# data/loader.py
# Downloads MNIST and splits it into non-overlapping subsets,
# one DataLoader per FL client, plus a shared test loader.

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def make_client_loaders(
    n_clients: int,
    samples_per_client: int = 500,
    batch_size: int = 32,
    data_dir: str = "./data",
) -> tuple[list[DataLoader], DataLoader]:
    """
    Returns:
        client_loaders : list of n_clients DataLoaders (training)
        test_loader    : single DataLoader over the full MNIST test set
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),   # MNIST mean/std
    ])

    train_dataset = datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
    test_dataset  = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    client_loaders = []
    for i in range(n_clients):
        start = i * samples_per_client
        end   = start + samples_per_client
        indices = list(range(start, end))
        subset  = Subset(train_dataset, indices)
        loader  = DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    return client_loaders, test_loader
