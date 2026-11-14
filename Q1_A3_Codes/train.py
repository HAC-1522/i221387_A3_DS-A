import argparse
import os
import torch
import torch.nn as nn
from torch.optim import SGD
from common import (build_model, get_loaders, set_seed, evaluate,
                    save_fp32_checkpoint, ARTIFACT_DIR)

def main():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 FP32 ResNet-18 baseline")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output", default=f"{ARTIFACT_DIR}/fp32_baseline.pt")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_loaders(args.batch_size, args.data_dir)
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(args.epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            correct += outputs.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
        scheduler.step()
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"loss={running_loss/total:.4f} | train_acc={100*correct/total:.2f}%")

    test_loss, test_acc = evaluate(model, test_loader, device)
    save_fp32_checkpoint(model.cpu(), args.output, args.epochs, test_acc)
    print(f"FP32 test loss={test_loss:.4f}, accuracy={test_acc:.2f}%")
    print(f"Saved checkpoint: {args.output}")

if __name__ == "__main__":
    main()
