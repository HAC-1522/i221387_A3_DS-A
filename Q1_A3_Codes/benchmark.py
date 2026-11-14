import argparse
import csv
import os
import time
import torch
from torch.utils.data import DataLoader
from common import build_model, get_datasets, CLASS_NAMES, get_backend

def load_fp32(path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    state = saved.get("state_dict", saved.get("model_state_dict", saved))
    model = build_model()
    model.load_state_dict(state)
    return model.eval()

def load_quantized(path):
    return torch.jit.load(
        path,
        map_location="cpu"
    ).eval()

def file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)

def main():
    parser = argparse.ArgumentParser(description="Benchmark FP32, PTQ and QAT models")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--fp32", default="artifacts/fp32_baseline.pt")
    parser.add_argument("--ptq", default="artifacts/ptq_int8.pt")
    parser.add_argument("--qat", default="artifacts/qat_int8.pt")
    parser.add_argument("--summary", default="benchmark_results.csv")
    parser.add_argument("--predictions", default="test_predictions.csv")
    args = parser.parse_args()

    torch.backends.quantized.engine = get_backend()
    for path in (args.fp32, args.ptq, args.qat):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required model checkpoint missing: {path}")

    _, test_ds = get_datasets()
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    models_to_test = [
        ("FP32", load_fp32(args.fp32), args.fp32),
        ("PTQ-INT8", load_quantized(args.ptq), args.ptq),
        ("QAT-INT8", load_quantized(args.qat), args.qat),
    ]

    summary_rows, prediction_rows = [], []
    for model_name, model, path in models_to_test:
        model.eval()
        correct, total = 0, 0
        total_loss = 0.0
        total_time = 0.0
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        with torch.inference_mode():
            for batch_index, (images, labels) in enumerate(loader):
                start = time.perf_counter()
                outputs = model(images)
                elapsed = time.perf_counter() - start
                total_time += elapsed
                loss = criterion(outputs, labels).item()
                probs = torch.softmax(outputs, dim=1)
                confidences, predictions = probs.max(dim=1)
                correct += predictions.eq(labels).sum().item()
                total_loss += loss
                total += labels.size(0)
                # One row per test image, with prediction and model-level latency.
                for j in range(labels.size(0)):
                    prediction_rows.append({
                        "model": model_name,
                        "sample_index": batch_index * args.batch_size + j,
                        "true_label": CLASS_NAMES[labels[j].item()],
                        "predicted_label": CLASS_NAMES[predictions[j].item()],
                        "confidence": f"{confidences[j].item():.6f}",
                    })

        accuracy = 100.0 * correct / total
        mean_ms = 1000.0 * total_time / total
        summary_rows.append({
            "model": model_name,
            "test_accuracy_percent": f"{accuracy:.4f}",
            "test_loss": f"{total_loss/total:.6f}",
            "mean_inference_ms_per_image": f"{mean_ms:.6f}",
            "checkpoint_size_mb": f"{file_size_mb(path):.4f}",
            "device": "cpu",
            "test_images": total,
        })
        for row in prediction_rows:
            if row["model"] == model_name:
                row["mean_inference_ms_per_image"] = f"{mean_ms:.6f}"
        print(summary_rows[-1])

    with open(args.summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    with open(args.predictions, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=prediction_rows[0].keys())
        writer.writeheader()
        writer.writerows(prediction_rows)

    print(f"Summary saved to: {args.summary}")
    print(f"Per-image predictions saved to: {args.predictions}")

if __name__ == "__main__":
    main()
