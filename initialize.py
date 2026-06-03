# TrashNet 垃圾分类模型推理脚本 - 2026.05

from torchvision import datasets
from   torch.utils.data import DataLoader

def dataset_initialize(dataset_path, transform_op=None, bs=32):

    name = None
    if "train" in dataset_path:
        names = "train"
    elif "val" in dataset_path:
        names = "val"
    elif "test" in dataset_path:
        names = "test"
    dataset = datasets.ImageFolder(
        dataset_path,
        transform=transform_op
    )
    class_names = dataset.classes

    print(f"Number of {names} samples: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

    return loader, class_names