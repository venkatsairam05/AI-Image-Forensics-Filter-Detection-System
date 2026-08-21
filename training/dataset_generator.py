import random
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw

from utils.config import DATASET_DIR, PRIMARY_CLASSES, FILTER_CLASSES, IMG_SIZE
from utils.logger import logger
from preprocessing.augmentation import apply_forensic_augmentations

class SyntheticDatasetGenerator:
    """
    Generates algorithmic baseline synthetic and natural sample images for zero-dependency
    local development, testing, and training initialization.
    """

    def __init__(self, root_dir: Path = DATASET_DIR, img_size: int = IMG_SIZE):
        self.root_dir = root_dir
        self.img_size = img_size

    def create_base_sample(self, seed: int = 42) -> Image.Image:
        """Generates a procedural photographic-like base scene."""
        np.random.seed(seed)
        # Create gradient sky/ground background
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        for y in range(self.img_size):
            r = int(120 + 80 * (y / self.img_size))
            g = int(140 + 60 * (y / self.img_size))
            b = int(200 - 100 * (y / self.img_size))
            img[y, :] = [r, g, b]

        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        
        # Add natural geometric objects
        for _ in range(3):
            x1 = random.randint(20, self.img_size - 80)
            y1 = random.randint(20, self.img_size - 80)
            x2 = x1 + random.randint(30, 70)
            y2 = y1 + random.randint(30, 70)
            color = (random.randint(40, 220), random.randint(40, 220), random.randint(40, 220))
            draw.ellipse([x1, y1, x2, y2], fill=color, outline=(255, 255, 255))

        return pil_img

    def generate_all(self, samples_per_class: int = 10):
        """Builds procedural image datasets across all splits and classes."""
        logger.info(f"Generating procedural sample dataset ({samples_per_class} per class)...")
        
        splits = {
            "train": samples_per_class,
            "validation": max(2, samples_per_class // 3),
            "test": max(2, samples_per_class // 3)
        }

        seed_counter = 100
        for split, count in splits.items():
            for cls in PRIMARY_CLASSES:
                target_folder = self.root_dir / split / cls.lower()
                target_folder.mkdir(parents=True, exist_ok=True)

                for i in range(count):
                    base = self.create_base_sample(seed=seed_counter)
                    seed_counter += 1

                    if cls == "REAL":
                        # Add natural camera sensor noise
                        sample = apply_forensic_augmentations(base, mode="gaussian_noise")
                    elif cls == "AI_GENERATED":
                        # Add smooth bilateral texture + high frequency sharpness
                        sample = apply_forensic_augmentations(base, mode="skin_smoothing")
                        sample = apply_forensic_augmentations(sample, mode="sharpening")
                    elif cls == "AI_EDITED":
                        sample = apply_forensic_augmentations(base, mode="blur")
                    elif cls == "FILTERED":
                        sample = apply_forensic_augmentations(base, mode="color_grading")
                    elif cls == "MANIPULATED":
                        sample = apply_forensic_augmentations(base, mode="jpeg_compression")
                        sample = apply_forensic_augmentations(sample, mode="sharpening")
                    else:
                        sample = base

                    out_path = target_folder / f"sample_{i:03d}.png"
                    sample.save(out_path, format="PNG")

        logger.info("Sample dataset generation completed successfully.")

if __name__ == "__main__":
    generator = SyntheticDatasetGenerator()
    generator.generate_all(samples_per_class=10)
