import argparse
import copy
import os
import torch
from torch.utils.data import DataLoader, Subset
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import get_default_qconfig_mapping
from common import (build_model, get_datasets, get_backend, load_fp32_checkpoint,
                    evaluate, ARTIFACT_DIR)

def main():
    parser = argparse.ArgumentParser(description="Post-Training Static Quantization (FX)")
    parser.add_argument("--checkpoint", default=f"{ARTIFACT_DIR}/fp32_baseline.pt")
    parser.add_argument("--calibration-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", default=f"{ARTIFACT_DIR}/ptq_int8.pt")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"FP32 checkpoint not found: {args.checkpoint}")

    backend = get_backend()
    torch.backends.quantized.engine = backend
    model = load_fp32_checkpoint(args.checkpoint, "cpu").eval()

    train_ds, test_ds = get_datasets()
    count = min(args.calibration_samples, len(train_ds))
    calibration_ds = Subset(train_ds, range(count))
    calibration_loader = DataLoader(calibration_ds, batch_size=args.batch_size,
                                    shuffle=False, num_workers=0)

    example_inputs = (torch.randn(1, 3, 32, 32),)
    qconfig_mapping = get_default_qconfig_mapping(backend)

    # FX graph-mode preparation performs graph-based operator fusion and inserts observers.
    prepared = prepare_fx(copy.deepcopy(model), qconfig_mapping, example_inputs)
    prepared.eval()
    with torch.inference_mode():
        for images, _ in calibration_loader:
            prepared(images)

    quantized = convert_fx(prepared).eval()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    example_input = torch.randn(1, 3, 32, 32)

    scripted_model = torch.jit.trace(
        quantized,
        example_input,
        check_trace=False
    )

    torch.jit.save(scripted_model, args.output)

    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)
    loss, accuracy = evaluate(quantized, test_loader, "cpu")
    print(f"Backend: {backend}")
    print(f"Calibration samples: {count}")
    print(f"PTQ-INT8 test loss={loss:.4f}, accuracy={accuracy:.2f}%")
    print(f"Saved quantized model: {args.output}")

if __name__ == "__main__":
    main()
