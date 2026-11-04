"""Train the repo's EfficientNet-B0 AI detector on REAL photographs vs
real synthetic images, instead of the procedural gradient/ellipse images
the shipped weights were originally trained on.

Reference training data used for the shipped 2026-11-04 weights:
SID-Set validation rows (saberzl/SID_Set, CC BY 4.0; OpenImages V7
real photos + FLUX full-synthetic images), 400 real + 400 synthetic
for training and 100 + 100 held out for evaluation. Training images
are NOT committed to this repo; place your own folders with the same
layout (train/real, train/ai_generated, test/real, test/ai_generated)
and pass --train-dir/--test-dir.

Labels follow utils.config.PRIMARY_CLASSES:
  REAL -> 0, AI_GENERATED -> 1. AI_EDITED/FILTERED/MANIPULATED never
  appear in this training set, so their logits are pushed down and the
  softmax becomes a clean Real vs AI_GENERATED split, which is what the
  demo's verdict needs. The filter detector (separate model) still
  handles filter labels.

Usage:  .venv/bin/python train_sid.py [--epochs-head 5] [--epochs-full 3]
Output: models_weights/ai_detector_sid.pth  (state_dict + val metrics)
"""
import argparse, json, sys, time
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import PRIMARY_CLASS_TO_IDX
from preprocessing.augmentation import get_train_transforms, get_val_transforms
from models.ai_detector import AIDetectionNetwork

WORK = Path(__file__).resolve().parent.parent

class FolderDataset(Dataset):
    def __init__(self, root, transform):
        self.samples = []
        for cls, idx in [("real", 0), ("ai_generated", 1)]:
            for p in sorted((Path(root) / cls).glob("*.jpg")):
                self.samples.append((p, idx))
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p, y = self.samples[i]
        return self.transform(Image.open(p).convert("RGB")), y

def evaluate(model, loader, device):
    model.eval(); correct = 0; total = 0
    per = {0: [0, 0], 1: [0, 0]}  # label -> [correct, total]
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item(); total += len(y)
            for t, pr in zip(y.tolist(), pred.tolist()):
                per[t][1] += 1
                if t == pr: per[t][0] += 1
    return correct / max(1, total), {k: round(v[0]/max(1, v[1]), 4) for k, v in per.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-head", type=int, default=6)
    ap.add_argument("--epochs-full", type=int, default=3)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-full", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--train-dir", type=Path, default=None)
    ap.add_argument("--test-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    device = "cpu"
    train_ds = FolderDataset(args.train_dir or WORK / "sid_train", get_train_transforms())
    test_ds = FolderDataset(args.test_dir or WORK / "sid_test", get_val_transforms())
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=0)
    print(f"train={len(train_ds)} test={len(test_ds)}", flush=True)

    model = AIDetectionNetwork().to(device)
    # Phase 1: frozen backbone, classifier head only
    for p in model.features.parameters(): p.requires_grad = False
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr_head, weight_decay=1e-3)
    crit = nn.CrossEntropyLoss()
    for epoch in range(1, args.epochs_head + 1):
        t0 = time.time(); model.train(); loss_sum = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); logits, _ = model(x); loss = crit(logits, y)
            loss.backward(); opt.step(); loss_sum += loss.item() * len(x)
        acc, per = evaluate(model, test_loader, device)
        print(f"[head] epoch {epoch} loss={loss_sum/len(train_ds):.4f} heldout_acc={acc:.4f} per_class={per} {time.time()-t0:.0f}s", flush=True)

    # Phase 2: unfreeze backbone at low LR
    for p in model.features.parameters(): p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_full, weight_decay=1e-3)
    for epoch in range(1, args.epochs_full + 1):
        t0 = time.time(); model.train(); loss_sum = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); logits, _ = model(x); loss = crit(logits, y)
            loss.backward(); opt.step(); loss_sum += loss.item() * len(x)
        acc, per = evaluate(model, test_loader, device)
        print(f"[full] epoch {epoch} loss={loss_sum/len(train_ds):.4f} heldout_acc={acc:.4f} per_class={per} {time.time()-t0:.0f}s", flush=True)

    out = args.out or (WORK / "models_weights" / "ai_detector_sid.pth")
    torch.save({"model_state_dict": model.state_dict(),
                "heldout_acc": acc, "per_class": per,
                "trained_on": "SID-Set validation (OpenImages real / FLUX synthetic), 400+400 train, 100+100 held-out"},
               out)
    print("saved", out, flush=True)

if __name__ == "__main__":
    main()
