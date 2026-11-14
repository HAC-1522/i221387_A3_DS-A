import os, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

DATA_DIR = os.environ.get("CIFAR10_DIR", "./data")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def build_model(pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    # CIFAR-10 is 32x32: use a CIFAR-friendly stem.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

def loaders(batch_size=128, workers=2):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_ds = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_tf)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers,
                       pin_memory=torch.cuda.is_available()),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                       pin_memory=torch.cuda.is_available()))

def evaluate(model, loader, device=DEVICE):
    model.eval(); correct = total = 0; loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x); loss_sum += criterion(out, y).item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item(); total += y.size(0)
    return loss_sum / total, 100.0 * correct / total

def train_epochs(model, loader, epochs, lr, device=DEVICE, save_path=None):
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train(); running = correct = total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x); loss = criterion(logits, y)
            loss.backward(); optimizer.step()
            running += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item(); total += y.size(0)
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}: loss={running/total:.4f}, train_acc={100*correct/total:.2f}%")
    if save_path:
        torch.save(model, save_path)
    return model
