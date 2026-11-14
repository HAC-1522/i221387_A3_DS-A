import argparse, os, copy, torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch_pruning as tp
from common import *

def load_model(path):
    # These are locally produced model artifacts, not untrusted checkpoints.
    obj=torch.load(path,map_location="cpu",weights_only=False)
    return obj["model"] if isinstance(obj,dict) and "model" in obj else obj

def unstructured_prune(model, amount=0.20, iterations=4):
    """Iterative global L1-magnitude pruning; remove masks to make zeros permanent."""
    layers=[m for m in model.modules() if isinstance(m,(nn.Conv2d,nn.Linear))]
    params=[(m,"weight") for m in layers]
    per_step=1.0-(1.0-amount)**(1.0/iterations)
    for i in range(iterations):
        prune.global_unstructured(params, pruning_method=prune.L1Unstructured,
                                  amount=per_step)
        print(f"Unstructured pruning iteration {i+1}/{iterations}")
    for module in layers:
        if hasattr(module,"weight_orig"):
            prune.remove(module,"weight")
    return model

def structured_prune(model, amount=0.25, iterations=4):
    """Dependency-aware channel pruning; torch-pruning updates residual dependencies."""
    example=torch.randn(1,3,32,32)
    ignored=[model.fc]
    importance=tp.importance.GroupMagnitudeImportance(p=2)
    pruner=tp.pruner.MagnitudePruner(
        model, example_inputs=example, importance=importance,
        pruning_ratio=amount, iterative_steps=iterations,
        ignored_layers=ignored, round_to=1)
    for i in range(iterations):
        pruner.step()
        print(f"Structured channel-pruning iteration {i+1}/{iterations}")
    return model

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",default="artifacts/original_fp32.pt")
    p.add_argument("--method",choices=["unstructured","structured"],required=True)
    p.add_argument("--amount",type=float,default=0.25)
    p.add_argument("--iterations",type=int,default=4)
    p.add_argument("--finetune-epochs",type=int,default=5)
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--output",default=None)
    a=p.parse_args()
    seed_everything(); os.makedirs("artifacts",exist_ok=True)
    model=load_model(a.input).to(DEVICE)
    if a.method=="unstructured": model=unstructured_prune(model,a.amount,a.iterations)
    else: model=structured_prune(model,a.amount,a.iterations)
    train_loader,test_loader=loaders(a.batch_size)
    before_loss,before_acc=evaluate(model,test_loader)
    print(f"After pruning, before fine-tuning: loss={before_loss:.4f}, accuracy={before_acc:.2f}%")
    train_epochs(model,train_loader,a.finetune_epochs,lr=0.001)
    loss,acc=evaluate(model,test_loader)
    out=a.output or f"artifacts/{a.method}_pruned.pt"
    torch.save(model,out)
    torch.save({"method":a.method,"test_loss":loss,"test_accuracy":acc,
                "amount":a.amount,"iterations":a.iterations},out.replace(".pt","_metadata.pt"))
    print(f"Saved {out}; fine-tuned test accuracy={acc:.2f}%")
if __name__=="__main__": main()
