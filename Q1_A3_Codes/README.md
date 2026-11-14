# Q1: PyTorch Model Quantization Framework (CIFAR-10)

This project trains a CIFAR-10 ResNet-18 FP32 baseline and produces two CPU INT8 models:
Post-Training Static Quantization (PTQ) and Quantization-Aware Training (QAT).

## 1. Environment

Recommended: Python 3.10–3.12 and a CPU PyTorch installation. From PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PyTorch is already installed and working in the environment used for Q2, keep that environment and install only missing dependencies.

## 2. Train FP32 baseline

```powershell
python train.py --epochs 10 --batch-size 128
```

Output: `artifacts/fp32_baseline.pt`

## 3. PTQ: calibration and INT8 conversion

```powershell
python ptq.py --calibration-samples 1024 --batch-size 128
```

Output: `artifacts/ptq_int8.pt`

The FX graph-mode preparation identifies/fuses supported operator patterns and inserts observers. Calibration runs representative training images through the prepared model; conversion replaces supported operations with quantized implementations.

## 4. QAT: fake quantization, fine-tuning and conversion

```powershell
python qat.py --epochs 5 --batch-size 128
```

Output: `artifacts/qat_int8.pt`

QAT starts from the FP32 checkpoint, prepares the model with fake-quantization modules, fine-tunes it, and converts it to an INT8 model.

## 5. Run one-image inference

```powershell
python inference.py --type fp32 --model artifacts/fp32_baseline.pt --image sample.png
python inference.py --type ptq --model artifacts/ptq_int8.pt --image sample.png
python inference.py --type qat --model artifacts/qat_int8.pt --image sample.png
```

Use a CIFAR-10-sized RGB image for the most representative result.

## 6. Benchmark all three models

```powershell
python benchmark.py
```

Outputs:
- `benchmark_results.csv`: one row per model with accuracy, loss, mean inference time, checkpoint size and device.
- `test_predictions.csv`: one row per test image per model with ground truth, predicted class and confidence.

## Execution order

Run steps 2, 3, 4, 5 (optional spot-check), then 6. PTQ and QAT both require the FP32 checkpoint.

## Technical notes

- The model is a CIFAR-10-adapted ResNet-18: 3x3 stride-1 input convolution, no max-pooling, and a 10-class output layer.
- Quantization is performed with PyTorch FX graph-mode static quantization APIs.
- The quantized models are intended for CPU inference. The script selects an available quantized CPU backend (`x86`, `fbgemm`, or `qnnpack`).
- PTQ calibration uses a deterministic prefix subset of the CIFAR-10 training dataset. The training transform includes random augmentation; for strict repeatability, set the seed and consider a deterministic calibration transform.
- Latency depends on CPU, PyTorch build, thread count, batch size and system load. Compare all models on the same machine and run.
- Checkpoint file size is the actual serialized file size, not theoretical parameter storage.
- Quantization APIs can vary between PyTorch releases. If `prepare_fx` or `prepare_qat_fx` is unavailable in an installed release, use a compatible PyTorch/torchvision pair rather than mixing package versions.


## 7. Verified Experimental Results

The framework was successfully executed in Google Colab using Python 3.13 and PyTorch 2.14.1+cu130. A Tesla T4 GPU was used for training, while all three models were benchmarked using CPU inference.

| Metric | FP32 | PTQ-INT8 | QAT-INT8 |
|---|---:|---:|---:|
| Test Accuracy | 84.90% | 85.00% | 85.35% |
| Test Loss | 0.440963 | 0.441982 | 0.434722 |
| Mean Inference (ms/image) | 13.093235 | 8.668671 | 8.739672 |
| Checkpoint Size (MB) | 42.7001 | 10.8489 | 10.8489 |

All models were evaluated on 10,000 CIFAR-10 test images.

### Results Analysis

- PTQ reduced checkpoint size by approximately 74.59% compared with FP32.
- QAT achieved the highest test accuracy of 85.35%.
- PTQ achieved the lowest measured inference latency of approximately 8.67 ms per image.
- Both INT8 models maintained comparable accuracy to the FP32 baseline while substantially reducing storage requirements.

## 8. Checkpoint Serialization

The FP32 model is stored as a PyTorch checkpoint containing its state dictionary. The converted PTQ and QAT models are exported as TorchScript checkpoints using `torch.jit.trace()` and `torch.jit.save()`.

The inference and benchmarking scripts load these quantized checkpoints using `torch.jit.load()`.

## 9. Execution Warnings

The tested PyTorch version generated deprecation warnings for FX graph-mode quantization and TorchScript serialization APIs. These warnings did not prevent training, conversion, checkpoint loading, inference, or benchmarking. Future PyTorch versions may require migration to newer quantization and export APIs.
