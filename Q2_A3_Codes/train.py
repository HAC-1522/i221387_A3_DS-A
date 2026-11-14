import argparse, os, torch
from common import *

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--no-pretrained", action="store_true",
                   help="Initialize ResNet-18 randomly instead of ImageNet pretrained weights.")
    p.add_argument("--output", default="artifacts/original_fp32.pt")
    a=p.parse_args()
    seed_everything(); os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    train_loader, test_loader=loaders(a.batch_size,a.workers)
    model=build_model(pretrained=not a.no_pretrained)
    model=train_epochs(model,train_loader,a.epochs,a.lr,save_path=a.output)
    loss,acc=evaluate(model,test_loader)
    print(f"Original model: test_loss={loss:.4f}, test_accuracy={acc:.2f}%")
    torch.save({"model":model,"test_accuracy":acc}, "artifacts/original_metadata.pt")
if __name__=="__main__": main()
