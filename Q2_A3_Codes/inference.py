import argparse, torch
from PIL import Image
from torchvision import transforms
from common import DEVICE

CLASSES=("airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck")
def load_model(path):
    model=torch.load(path,map_location=DEVICE,weights_only=False)
    return model.to(DEVICE).eval()

def predict(model,image_path):
    tf=transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
    image=Image.open(image_path).convert("RGB")
    x=tf(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        probs=model(x).softmax(1)[0]
    score,index=probs.max(0)
    return CLASSES[index.item()],float(score.item())

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True,help="Path to an exported .pt model")
    p.add_argument("--image",required=True)
    a=p.parse_args()
    model=load_model(a.model); label,confidence=predict(model,a.image)
    print(f"Prediction: {label} | confidence: {confidence:.4f}")
if __name__=="__main__": main()
