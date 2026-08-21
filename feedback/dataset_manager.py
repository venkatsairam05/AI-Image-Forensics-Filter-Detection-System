import shutil
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple
from PIL import Image

from utils.config import DATASET_DIR, PRIMARY_CLASSES, FILTER_CLASSES
from utils.logger import logger
from preprocessing.image_processor import ImageProcessor

class DatasetManager:
    """
    Manages dataset directory layout, ingestion of verified samples,
    sanitization safeguards against malicious/corrupted images, and class balancing.
    """

    def __init__(self, root_dir: Path = DATASET_DIR):
        self.root_dir = root_dir
        self.processor = ImageProcessor()
        self.splits = ["train", "validation", "test"]
        self.init_structure()

    def init_structure(self):
        """Initializes primary and filter dataset directories."""
        for split in self.splits:
            for cls in PRIMARY_CLASSES:
                d = self.root_dir / split / cls.lower()
                d.mkdir(parents=True, exist_ok=True)

        filters_dir = self.root_dir / "filters"
        for flt in FILTER_CLASSES:
            (filters_dir / flt).mkdir(parents=True, exist_ok=True)

    def get_dataset_stats(self) -> Dict[str, Any]:
        """Calculates image distribution across all splits and classes."""
        stats: Dict[str, Any] = {
            "total_images": 0,
            "splits": {}
        }

        for split in self.splits:
            split_stats = {}
            split_total = 0
            for cls in PRIMARY_CLASSES:
                folder = self.root_dir / split / cls.lower()
                count = len(list(folder.glob("*.*"))) if folder.exists() else 0
                split_stats[cls] = count
                split_total += count
            stats["splits"][split] = {
                "counts": split_stats,
                "total": split_total
            }
            stats["total_images"] += split_total

        return stats

    def ingest_sample(self, src_path: Path, target_class: str, split: str = "train") -> bool:
        """
        Safely copies a verified image into the dataset with validation and de-duplication.
        """
        target_class_norm = target_class.upper()
        if target_class_norm not in PRIMARY_CLASSES:
            logger.error(f"Invalid target class: {target_class}")
            return False

        if not src_path.exists():
            logger.error(f"Source file does not exist: {src_path}")
            return False

        # Validate image integrity
        valid, msg, pil_img = self.processor.validate_image(src_path)
        if not valid or pil_img is None:
            logger.warning(f"Rejected corrupted or invalid sample: {msg}")
            return False

        # Compute hash for de-duplication
        h = self.processor.compute_sha256(pil_img)
        dest_dir = self.root_dir / split / target_class_norm.lower()
        dest_file = dest_dir / f"{h[:16]}.png"

        if dest_file.exists():
            logger.info(f"Duplicate sample already in dataset: {dest_file.name}")
            return True

        pil_img.save(dest_file, format="PNG")
        logger.info(f"Ingested sample into {split}/{target_class_norm.lower()}: {dest_file.name}")
        return True
