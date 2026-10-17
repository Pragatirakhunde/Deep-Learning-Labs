"""
DL Practical 01: Small-dataset image classification using pretrained ViT vs CNN.
Uses scikit-learn's built-in 8x8 Digits dataset (about 1,800 images; no dataset download).
Pretrained ImageNet model weights are downloaded separately on the first run.
"""
import argparse, csv, os, random, time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
from sklearn.datasets import load_digits

class DigitsImageDataset(Dataset):
    def __init__(self, indices, images, labels, transform):
        self.indices, self.images, self.labels, self.transform = indices, images, labels, transform
    def __len__(self): return len(self.indices)
    def __getitem__(self, item):
        idx = self.indices[item]
        # Digits are 8x8 grayscale; convert to RGB for ImageNet-pretrained models.
        image = Image.fromarray((self.images[idx] * 16).clip(0, 255).astype(np.uint8)).convert("RGB")
        return self.transform(image), int(self.labels[idx])

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def build_model(name, num_classes, device):
    if name == "vit":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    for p in model.parameters(): p.requires_grad = False
    head = model.heads.head if name == "vit" else model.fc
    for p in head.parameters(): p.requires_grad = True
    return model.to(device)

def evaluate(model, loader, device):
    model.eval(); correct = total = 0; loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x); loss = criterion(out, y)
            loss_sum += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item(); total += y.size(0)
    return 100 * correct / max(total, 1), loss_sum / max(total, 1)

def train_one(name, train_loader, test_loader, epochs, lr, device):
    print(f"\\n{'='*16} {name.upper()} {'='*16}")
    model = build_model(name, 10, device)
    head = model.heads.head if name == "vit" else model.fc
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = []; start = time.perf_counter()
    for epoch in range(epochs):
        model.train(); running = seen = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(); out = model(x); loss = criterion(out, y)
            loss.backward(); optimizer.step()
            running += loss.item() * y.size(0); seen += y.size(0)
        acc, test_loss = evaluate(model, test_loader, device)
        row = {"model": name.upper(), "epoch": epoch+1,
               "train_loss": running/max(seen,1), "test_loss": test_loss,
               "test_accuracy_percent": acc}
        history.append(row)
        print(f"Epoch {epoch+1}/{epochs} | train loss {row['train_loss']:.4f} "
              f"| test loss {test_loss:.4f} | test accuracy {acc:.2f}%")
    elapsed = time.perf_counter() - start
    acc, loss = evaluate(model, test_loader, device)
    return history, {"model": name.upper(), "test_accuracy_percent": round(acc,2),
                     "test_loss": round(loss,4), "training_seconds": round(elapsed,2)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-samples", type=int, default=800)
    parser.add_argument("--test-samples", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--models", choices=["both","vit","cnn"], default="both")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if min(args.epochs,args.train_samples,args.test_samples,args.batch_size) < 1:
        parser.error("All counts must be positive.")
    os.makedirs("outputs", exist_ok=True); set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Dataset: built-in sklearn Digits (no separate dataset download).")
    print("First run downloads pretrained model weights only.")
    transform = transforms.Compose([
        transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])
    digits = load_digits()
    n = len(digits.target)
    indices = np.arange(n); rng = np.random.default_rng(42); rng.shuffle(indices)
    n_test = min(args.test_samples, max(1, int(0.2*n)))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    train_idx = train_idx[:min(args.train_samples, len(train_idx))]
    test_ds = DigitsImageDataset(test_idx, digits.images, digits.target, transform)
    train_ds = DigitsImageDataset(train_idx, digits.images, digits.target, transform)
    train_loader = DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.workers)
    test_loader = DataLoader(test_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.workers)
    chosen = ["vit","cnn"] if args.models == "both" else [args.models]
    history_all=[]; summaries=[]
    for name in chosen:
        history, summary = train_one(name,train_loader,test_loader,args.epochs,1e-3,device)
        history_all.extend(history); summaries.append(summary)
    with open("outputs/comparison_results.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0].keys())); w.writeheader(); w.writerows(summaries)
    with open("outputs/training_history.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(history_all[0].keys())); w.writeheader(); w.writerows(history_all)
    labels=[s["model"] for s in summaries]; accs=[s["test_accuracy_percent"] for s in summaries]
    times=[s["training_seconds"] for s in summaries]
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    axes[0].bar(labels,accs); axes[0].set_title("Test accuracy"); axes[0].set_ylabel("Accuracy (%)")
    axes[1].bar(labels,times); axes[1].set_title("Training time"); axes[1].set_ylabel("Seconds")
    fig.tight_layout(); fig.savefig("outputs/comparison_plot.png",dpi=160); plt.close(fig)
    print("\\nFINAL RESULTS")
    for s in summaries: print(s)
    print("Saved results in outputs/.")
if __name__=="__main__": main()
