
import argparse
import copy
import os
import torch
import torch.nn as nn
from torch.optim import SGD

from torch.ao.quantization.quantize_fx import (
    prepare_qat_fx,
    convert_fx
)
from torch.ao.quantization import get_default_qat_qconfig_mapping

from common import (
    get_loaders,
    get_backend,
    load_fp32_checkpoint,
    evaluate,
    ARTIFACT_DIR
)


def main():
    parser = argparse.ArgumentParser(
        description="Quantization-Aware Training (FX)"
    )

    parser.add_argument(
        "--checkpoint",
        default=f"{ARTIFACT_DIR}/fp32_baseline.pt"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument(
        "--output",
        default=f"{ARTIFACT_DIR}/qat_int8.pt"
    )

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"FP32 checkpoint not found: {args.checkpoint}"
        )

    # GPU is used for QAT training.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Quantized inference and conversion use CPU.
    backend = get_backend()
    torch.backends.quantized.engine = backend

    print(f"Training device: {device}")
    print(f"Quantization backend: {backend}")

    train_loader, test_loader = get_loaders(args.batch_size)

    # Load the pretrained FP32 model on CPU.
    model = load_fp32_checkpoint(args.checkpoint, "cpu")
    model.train()

    example_inputs = (torch.randn(1, 3, 32, 32),)

    qconfig_mapping = get_default_qat_qconfig_mapping(backend)

    # Insert fake quantization and observer modules.
    qat_model = prepare_qat_fx(
        copy.deepcopy(model),
        qconfig_mapping,
        example_inputs
    )

    # Move prepared model to GPU for QAT fine-tuning.
    qat_model = qat_model.to(device)
    qat_model.train()

    criterion = nn.CrossEntropyLoss()

    optimizer = SGD(
        qat_model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    # QAT fine-tuning.
    for epoch in range(args.epochs):
        qat_model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = qat_model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += outputs.argmax(1).eq(labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        print(
            f"QAT epoch {epoch + 1}/{args.epochs} | "
            f"loss={running_loss / total:.4f} | "
            f"train_acc={100 * correct / total:.2f}%"
        )

    # Move model to CPU before INT8 conversion.
    qat_model = qat_model.to("cpu")
    qat_model.eval()

    quantized = convert_fx(qat_model).eval()

    # Save final INT8 model.
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    example_input = torch.randn(1, 3, 32, 32)

    scripted_model = torch.jit.trace(
        quantized,
        example_input,
        check_trace=False
    )

    torch.jit.save(scripted_model, args.output)

    # Evaluate quantized model on CPU.
    loss, accuracy = evaluate(
        quantized,
        test_loader,
        "cpu"
    )

    print(f"Backend: {backend}")
    print(f"QAT-INT8 test loss={loss:.4f}")
    print(f"QAT-INT8 test accuracy={accuracy:.2f}%")
    print(f"Saved quantized model: {args.output}")


if __name__ == "__main__":
    main()
