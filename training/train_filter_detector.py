import time
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from utils.config import (
    DATASET_DIR, MODELS_WEIGHTS_DIR, FILTER_MODEL_PATH,
    FILTER_CLASSES, DEVICE, NUM_WORKERS
)
from utils.logger import logger
from preprocessing.augmentation import get_train_transforms, get_val_transforms
from models.filter_detector import FilterDetectionNetwork

class MultiLabelFilterDataset(Dataset):
    """Dataset simulating multi-label image filter combinations."""

    def __init__(self, split_dir: Path, transform=None):
        self.samples = []
        self.transform = transform
        
        # Collect all images across splits
        all_imgs = list(split_dir.glob("**/*.*"))
        for p in all_imgs:
            if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
                # Generate deterministic synthetic multi-label vector based on parent directory
                parent_name = p.parent.name.lower()
                target_vec = np.zeros(len(FILTER_CLASSES), dtype=np.float32)
                
                if "filter" in parent_name:
                    target_vec[0] = 1.0 # beauty
                    target_vec[2] = 1.0 # color_grading
                if "ai_gen" in parent_name:
                    target_vec[1] = 1.0 # smoothing
                    target_vec[8] = 1.0 # upscaling
                if "manip" in parent_name:
                    target_vec[4] = 1.0 # sharpening
                    target_vec[9] = 1.0 # compression
                
                self.samples.append((p, target_vec))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_vec = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label_vec, dtype=torch.float32), str(path)


def train_filter_detector(
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-4,
    device: str = DEVICE
) -> Dict[str, Any]:
    """Trains multi-label filter detection model."""
    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "validation"

    train_dataset = MultiLabelFilterDataset(train_dir, transform=get_train_transforms())
    val_dataset = MultiLabelFilterDataset(val_dir, transform=get_val_transforms())

    if len(train_dataset) == 0:
        from training.dataset_generator import SyntheticDatasetGenerator
        SyntheticDatasetGenerator().generate_all(samples_per_class=8)
        train_dataset = MultiLabelFilterDataset(train_dir, transform=get_train_transforms())
        val_dataset = MultiLabelFilterDataset(val_dir, transform=get_val_transforms())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    model = FilterDetectionNetwork().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    logger.info(f"Beginning Multi-Label Filter Detector training on {device}...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_train_loss = running_loss / max(1, len(train_dataset))

        # Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)

        epoch_val_loss = val_loss / max(1, len(val_dataset))
        logger.info(f"Filter Epoch [{epoch}/{epochs}] - Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

    FILTER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "final_val_loss": epoch_val_loss
    }, FILTER_MODEL_PATH)

    logger.info(f"Filter Detector training finished in {time.time() - start_time:.1f}s. Saved to {FILTER_MODEL_PATH.name}")
    return {"status": "success", "model_path": str(FILTER_MODEL_PATH)}

if __name__ == "__main__":
    train_filter_detector(epochs=3)
