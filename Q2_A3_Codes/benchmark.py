import argparse, csv, os, time, torch
from common import *

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def benchmark_one(name,path,loader,device,warmup=10):
    model=torch.load(path,map_location=device,weights_only=False).to(device).eval()
    criterion=torch.nn.CrossEntropyLoss()
    correct=total=0; loss_sum=0.0; batch_times=[]
    with torch.inference_mode():
        for i,(x,y) in enumerate(loader):
            x,y=x.to(device),y.to(device)
            if device.type=="cuda": torch.cuda.synchronize()
            start=time.perf_counter(); out=model(x)
            if device.type=="cuda": torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            if i>=warmup: batch_times.append(elapsed/x.size(0))
            loss_sum+=criterion(out,y).item()*y.size(0)
            correct+=(out.argmax(1)==y).sum().item(); total+=y.size(0)
    avg_ms=(sum(batch_times)/len(batch_times)*1000) if batch_times else float("nan")
    params=count_parameters(model)
    return {"model":name,"test_accuracy_percent":100*correct/total,
            "test_loss":loss_sum/total,"parameters":params,
            "parameter_size_mb_fp32":params*4/(1024**2),
            "mean_inference_ms_per_image":avg_ms,
            "device":str(device)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--original",default="artifacts/original_fp32.pt")
    p.add_argument("--unstructured",default="artifacts/unstructured_pruned.pt")
    p.add_argument("--structured",default="artifacts/structured_pruned.pt")
    p.add_argument("--output",default="benchmark_results.csv")
    p.add_argument("--batch-size",type=int,default=128)
    a=p.parse_args()
    _,test_loader=loaders(a.batch_size)
    rows=[]
    for name,path in [("original",a.original),("unstructured",a.unstructured),
                      ("structured",a.structured)]:
        if not os.path.exists(path):
            print(f"Skipping {name}: missing {path}"); continue
        row=benchmark_one(name,path,test_loader,DEVICE); rows.append(row)
        print(row)
    if not rows: raise RuntimeError("No model artifacts found.")
    with open(a.output,"w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    print(f"Benchmark CSV written to {a.output}")
if __name__=="__main__": main()
