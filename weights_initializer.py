import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from utils.config import AI_MODEL_PATH, FILTER_MODEL_PATH, MODELS_WEIGHTS_DIR
from utils.logger import logger
from training.dataset_generator import SyntheticDatasetGenerator
from training.train_ai_detector import train_ai_detector
from training.train_filter_detector import train_filter_detector

def initialize_system_weights():
    """
    Bootstraps the project dataset and trains calibrated baseline weights for both
    the AI Generation Detector and Multi-Label Filter Detector.
    """
    logger.info("==================================================")
    logger.info("Initializing AI Image Forensics Model Weights...")
    logger.info("==================================================")

    # 1. Generate procedural datasets
    generator = SyntheticDatasetGenerator(img_size=256)
    generator.generate_all(samples_per_class=12)

    # 2. Train AI Generation Detector baseline
    logger.info("\n[Phase 1/2] Training AI Generation Classifier...")
    ai_result = train_ai_detector(epochs=3, batch_size=6, lr=1e-4)

    # 3. Train Filter Detector baseline
    logger.info("\n[Phase 2/2] Training Multi-Label Filter Classifier...")
    filter_result = train_filter_detector(epochs=3, batch_size=6, lr=1e-4)

    logger.info("\n==================================================")
    logger.info("SYSTEM INITIALIZATION COMPLETE")
    logger.info(f"AI Detector Model:     {AI_MODEL_PATH}")
    logger.info(f"Filter Detector Model: {FILTER_MODEL_PATH}")
    logger.info("==================================================")

if __name__ == "__main__":
    initialize_system_weights()
