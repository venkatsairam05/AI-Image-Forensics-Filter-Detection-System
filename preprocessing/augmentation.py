import random
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import torchvision.transforms as T
from utils.config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD

def get_train_transforms(img_size: int = IMG_SIZE) -> T.Compose:
    """Returns training data augmentations with geometric and photometric perturbations."""
    return T.Compose([
        T.Resize((int(img_size * 1.1), int(img_size * 1.1))),
        T.RandomCrop((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.1),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.RandomRotation(degrees=15),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def get_val_transforms(img_size: int = IMG_SIZE) -> T.Compose:
    """Returns standard validation/test transforms."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def apply_forensic_augmentations(img: Image.Image, mode: str = "random") -> Image.Image:
    """
    Applies synthetic computer vision transformations for testing and dataset generation:
    - JPEG compression degradation
    - Gaussian / Poisson / Salt&Pepper noise
    - High-frequency unsharp mask sharpening
    - Bilateral facial skin smoothing
    - Color grading LUT shift
    - HDR dynamic tone mapping
    """
    np_img = np.array(img)

    if mode == "jpeg_compression" or (mode == "random" and random.random() < 0.3):
        # Simulate severe JPEG artifacts
        quality = random.randint(30, 75)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        result, enc = cv2.imencode('.jpg', bgr, encode_param)
        dec = cv2.imdecode(enc, 1)
        np_img = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)

    elif mode == "skin_smoothing" or (mode == "random" and random.random() < 0.3):
        # Bilateral filter preserves sharp edges while smoothing skin texture
        bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        smoothed = cv2.bilateralFilter(bgr, d=9, sigmaColor=75, sigmaSpace=75)
        np_img = cv2.cvtColor(smoothed, cv2.COLOR_BGR2RGB)

    elif mode == "sharpening" or (mode == "random" and random.random() < 0.3):
        # Unsharp masking
        gaussian = cv2.GaussianBlur(np_img, (0, 0), 2.0)
        np_img = cv2.addWeighted(np_img, 1.5, gaussian, -0.5, 0)
        np_img = np.clip(np_img, 0, 255).astype(np.uint8)

    elif mode == "gaussian_noise" or (mode == "random" and random.random() < 0.3):
        # Additive Gaussian sensor noise
        row, col, ch = np_img.shape
        mean = 0
        sigma = random.uniform(10, 25)
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        noisy = np_img + gauss
        np_img = np.clip(noisy, 0, 255).astype(np.uint8)

    elif mode == "color_grading" or (mode == "random" and random.random() < 0.3):
        # Cinematic tone curve
        pil_tmp = Image.fromarray(np_img)
        enhancer = ImageEnhance.Color(pil_tmp)
        pil_tmp = enhancer.enhance(random.uniform(1.3, 1.8))
        contrast = ImageEnhance.Contrast(pil_tmp)
        pil_tmp = contrast.enhance(random.uniform(1.2, 1.6))
        np_img = np.array(pil_tmp)

    elif mode == "blur":
        np_img = cv2.GaussianBlur(np_img, (9, 9), 3.0)

    return Image.fromarray(np_img)
