import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

DATA_DIR = "./data"
ARTIFACT_DIR = "./artifacts"
NUM_CLASSES = 10
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def build_model():
    """CIFAR-10 adapted ResNet-18: 3x3 stem, stride 1, no max-pooling."""
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model

def get_transforms():
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    return train_tf, test_tf

def get_datasets(data_dir=DATA_DIR, download=True):
    train_tf, test_tf = get_transforms()
    train_ds = datasets.CIFAR10(root=data_dir, train=True, download=download, transform=train_tf)
    test_ds = datasets.CIFAR10(root=data_dir, train=False, download=download, transform=test_tf)
    return train_ds, test_ds

def get_loaders(batch_size=128, data_dir=DATA_DIR, workers=0):
    train_ds, test_ds = get_datasets(data_dir)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=workers, pin_memory=False)
    return train_loader, test_loader

def get_backend():
    supported = list(torch.backends.quantized.supported_engines)
    for candidate in ("x86", "fbgemm", "qnnpack"):
        if candidate in supported:
            return candidate
    raise RuntimeError(f"No supported quantized CPU backend found. Available: {supported}")

def evaluate(model, loader, device="cpu"):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            correct += outputs.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / total, 100.0 * correct / total

def save_fp32_checkpoint(model, path, epoch=None, accuracy=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "model_name": "cifar_resnet18",
        "num_classes": NUM_CLASSES,
        "epoch": epoch,
        "test_accuracy": accuracy,
    }, path)

def load_fp32_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    model = build_model()
    model.load_state_dict(state)
    return model
