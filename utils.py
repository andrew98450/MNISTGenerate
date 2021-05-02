import torchvision

def load_dataset(root="Dataset", download=True):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))])
    dataset = torchvision.datasets.MNIST(
        root=root,
        train=True,
        transform=transform,
        download=download)
    return dataset

    