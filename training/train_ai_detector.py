import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from utils.config import (
    DATASET_DIR, MODELS_WEIGHTS_DIR, AI_MODEL_PATH,
    PRIMARY_CLASSES, PRIMARY_CLASS_TO_IDX, DEVICE, NUM_WORKERS
)
from utils.logger import logger
from preprocessing.augmentation import get_train_transforms, get_val_transforms
from models.ai_detector import AIDetectionNetwork

class ImageClassificationDataset(Dataset):
    """PyTorch Dataset loading images from directory class folders."""

    def __init__(self, split_dir: Path, transform=None):
        self.samples = []
        self.transform = transform

        for cls_name in PRIMARY_CLASSES:
            cls_folder = split_dir / cls_name.lower()
            if not cls_folder.exists():
                continue
            cls_idx = PRIMARY_CLASS_TO_IDX[cls_name]
            for file_path in cls_folder.glob("*.*"):
                if file_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
                    self.samples.append((file_path, cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        img = Image.open(file_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, str(file_path)


def train_ai_detector(
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-4,
    device: str = DEVICE
) -> Dict[str, Any]:
    """Trains the primary AI detection network with transfer learning."""
    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "validation"

    train_dataset = ImageClassificationDataset(train_dir, transform=get_train_transforms())
    val_dataset = ImageClassificationDataset(val_dir, transform=get_val_transforms())

    if len(train_dataset) == 0:
        logger.warning("No training samples found in dataset/train! Generating procedural baseline samples...")
        from training.dataset_generator import SyntheticDatasetGenerator
        SyntheticDatasetGenerator().generate_all(samples_per_class=8)
        train_dataset = ImageClassificationDataset(train_dir, transform=get_train_transforms())
        val_dataset = ImageClassificationDataset(val_dir, transform=get_val_transforms())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    model = AIDetectionNetwork().to(device)
    criterion_primary = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    logger.info(f"Beginning AI Detector training for {epochs} epochs on {device}...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            primary_logits, _ = model(images)
            loss = criterion_primary(primary_logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        scheduler.step()
        epoch_train_loss = running_loss / max(1, len(train_dataset))

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                primary_logits, _ = model(images)
                loss = criterion_primary(primary_logits, labels)
                val_loss += loss.item() * images.size(0)

                preds = torch.argmax(primary_logits, dim=1)
                correct += (preds == labels).sum().item()

        epoch_val_loss = val_loss / max(1, len(val_dataset))
        epoch_val_acc = correct / max(1, len(val_dataset))

        history["train_loss"].append(round(epoch_train_loss, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["val_acc"].append(round(epoch_val_acc, 4))

        logger.info(f"Epoch [{epoch}/{epochs}] - Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%")

        # Save checkpoint if best
        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            AI_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": best_val_acc,
                "history": history
            }, AI_MODEL_PATH)

    # Save metrics history to JSON
    metrics_path = MODELS_WEIGHTS_DIR / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training completed in {time.time() - start_time:.1f}s. Best Val Accuracy: {best_val_acc*100:.2f}%. Model saved to {AI_MODEL_PATH.name}")
    return {
        "best_val_acc": best_val_acc,
        "history": history,
        "model_path": str(AI_MODEL_PATH)
    }

if __name__ == "__main__":
    train_ai_detector(epochs=3)
