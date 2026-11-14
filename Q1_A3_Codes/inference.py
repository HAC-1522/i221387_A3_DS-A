import argparse
import torch
from PIL import Image
from torchvision import transforms
from common import build_model, CLASS_NAMES, get_backend

def load_model(model_type, checkpoint):
    torch.backends.quantized.engine = get_backend()
    if model_type == "fp32":
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = saved.get("state_dict", saved.get("model_state_dict", saved))
        model = build_model()
        model.load_state_dict(state)
        return model.eval()
    # Load PTQ and QAT TorchScript INT8 checkpoints.
    return torch.jit.load(checkpoint, map_location="cpu").eval()

def main():
    parser = argparse.ArgumentParser(description="Run CIFAR-10 image inference")
    parser.add_argument("--type", choices=["fp32", "ptq", "qat"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    image = Image.open(args.image).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    model = load_model(args.type, args.model)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, index = probabilities.max(dim=0)
    print(f"Prediction: {CLASS_NAMES[index.item()]}")
    print(f"Confidence: {confidence.item():.4f}")

if __name__ == "__main__":
    main()
