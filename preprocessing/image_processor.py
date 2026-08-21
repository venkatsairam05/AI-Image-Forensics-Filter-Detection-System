import io
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import cv2
from PIL import Image, ExifTags, ImageOps
import torch
import torchvision.transforms as T

from utils.config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, DEVICE
from utils.logger import logger

class ImageProcessor:
    """Production image preprocessing, validation, hashing, and metadata extraction engine."""

    def __init__(self, target_size: int = IMG_SIZE):
        self.target_size = target_size
        self.tensor_transform = T.Compose([
            T.Resize((target_size, target_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def validate_image(self, file_source: Union[str, Path, bytes, io.BytesIO, Image.Image]) -> Tuple[bool, Optional[str], Optional[Image.Image]]:
        """
        Validates file integrity and readable image format.
        Supports JPG, JPEG, PNG, WEBP, BMP, TIFF.
        """
        try:
            if isinstance(file_source, Image.Image):
                img = file_source.copy()
            elif isinstance(file_source, (str, Path)):
                img = Image.open(file_source)
            elif isinstance(file_source, bytes):
                img = Image.open(io.BytesIO(file_source))
            elif isinstance(file_source, io.BytesIO):
                file_source.seek(0)
                img = Image.open(file_source)
            else:
                return False, "Unsupported file input type", None

            # Verify image
            img.verify()
            
            # Reopen for actual decoding (verify invalidates file pointer in PIL)
            if isinstance(file_source, (str, Path)):
                img = Image.open(file_source)
            elif isinstance(file_source, bytes):
                img = Image.open(io.BytesIO(file_source))
            elif isinstance(file_source, io.BytesIO):
                file_source.seek(0)
                img = Image.open(file_source)
            elif isinstance(file_source, Image.Image):
                img = file_source.copy()

            # Handle EXIF orientation
            try:
                img = ImageOps.exif_transpose(img)
            except Exception:
                pass

            # Convert to RGB
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Check dimensions
            w, h = img.size
            if w < 16 or h < 16:
                return False, f"Image dimensions too small ({w}x{h}). Minimum is 16x16.", None
            if w > 10000 or h > 10000:
                return False, f"Image dimensions excessively large ({w}x{h}). Maximum is 10000x10000.", None

            return True, "Valid image", img
        except Exception as e:
            logger.error(f"Image validation failed: {e}")
            return False, f"Invalid or corrupted image: {str(e)}", None

    def pil_to_cv2(self, pil_img: Image.Image) -> np.ndarray:
        """Converts PIL RGB Image to OpenCV BGR numpy array."""
        rgb_arr = np.array(pil_img)
        return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

    def cv2_to_pil(self, cv2_img: np.ndarray) -> Image.Image:
        """Converts OpenCV BGR numpy array to PIL RGB Image."""
        rgb_arr = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_arr)

    def extract_metadata(self, pil_img: Image.Image, raw_bytes: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Extracts EXIF metadata, ICC profiles, dimensions, software signatures, and SHA-256 hash.
        """
        w, h = pil_img.size
        metadata: Dict[str, Any] = {
            "width": w,
            "height": h,
            "aspect_ratio": round(w / max(1, h), 3),
            "mode": pil_img.mode,
            "format": pil_img.format if pil_img.format else "Unknown",
            "has_exif": False,
            "camera_make": "None / Stripped",
            "camera_model": "None / Stripped",
            "software": "None / Stripped",
            "date_time": "None",
            "flash": "None",
            "iso": "None",
            "focal_length": "None",
            "exposure_time": "None",
            "f_number": "None",
            "raw_exif": {},
            "hashes": {}
        }

        # Perceptual & Cryptographic Hashes
        metadata["hashes"]["sha256"] = self.compute_sha256(pil_img, raw_bytes)
        metadata["hashes"]["phash"] = self.compute_phash(pil_img)
        metadata["hashes"]["dhash"] = self.compute_dhash(pil_img)

        # Extract EXIF if available
        try:
            exif_data = pil_img.getexif()
            if exif_data and len(exif_data) > 0:
                metadata["has_exif"] = True
                parsed_exif = {}
                for tag_id, value in exif_data.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    if isinstance(value, bytes):
                        try:
                            value = value.decode("utf-8", errors="ignore").strip()
                        except Exception:
                            value = str(value)
                    parsed_exif[tag_name] = str(value)

                metadata["raw_exif"] = parsed_exif
                metadata["camera_make"] = parsed_exif.get("Make", "None / Stripped")
                metadata["camera_model"] = parsed_exif.get("Model", "None / Stripped")
                metadata["software"] = parsed_exif.get("Software", "None / Stripped")
                metadata["date_time"] = parsed_exif.get("DateTime", "None")
                metadata["flash"] = parsed_exif.get("Flash", "None")
                metadata["iso"] = parsed_exif.get("ISOSpeedRatings", "None")
                metadata["focal_length"] = parsed_exif.get("FocalLength", "None")
                metadata["exposure_time"] = parsed_exif.get("ExposureTime", "None")
                metadata["f_number"] = parsed_exif.get("FNumber", "None")
        except Exception as e:
            logger.warning(f"EXIF parsing warning: {e}")

        # Flag known AI generation software markers in metadata
        software_lower = str(metadata["software"]).lower()
        metadata["ai_software_indicator"] = any(
            marker in software_lower for marker in [
                "midjourney", "stable diffusion", "dall-e", "novelai",
                "automatic1111", "comfyui", "photoshop generative fill", "adobe firefly"
            ]
        )

        return metadata

    def compute_sha256(self, pil_img: Image.Image, raw_bytes: Optional[bytes] = None) -> str:
        """Computes SHA-256 hash of raw bytes or image pixel array."""
        if raw_bytes is not None and len(raw_bytes) > 0:
            return hashlib.sha256(raw_bytes).hexdigest()
        return hashlib.sha256(np.array(pil_img).tobytes()).hexdigest()

    def compute_phash(self, pil_img: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> str:
        """Computes 64-bit Perceptual Hash (pHash) via 2D Discrete Cosine Transform."""
        img = pil_img.convert("L").resize((hash_size * highfreq_factor, hash_size * highfreq_factor), Image.Resampling.BILINEAR)
        pixels = np.asarray(img, dtype=np.float32)
        dct = cv2.dct(pixels)
        dct_low = dct[:hash_size, :hash_size]
        med = np.median(dct_low)
        diff = dct_low > med
        return "".join(["1" if b else "0" for b in diff.flatten()])

    def compute_dhash(self, pil_img: Image.Image, hash_size: int = 8) -> str:
        """Computes Difference Hash (dHash) tracking horizontal intensity gradients."""
        img = pil_img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
        pixels = np.asarray(img, dtype=np.int32)
        diff = pixels[:, 1:] > pixels[:, :-1]
        return "".join(["1" if b else "0" for b in diff.flatten()])

    def to_tensor(self, pil_img: Image.Image, device: str = DEVICE) -> torch.Tensor:
        """Transforms PIL Image into a normalized 4D PyTorch tensor (1, 3, H, W)."""
        tensor = self.tensor_transform(pil_img)
        return tensor.unsqueeze(0).to(device)

    def crop_face(self, pil_img: Image.Image, box: Tuple[int, int, int, int], margin: float = 0.2) -> Image.Image:
        """Crops face box (x, y, w, h) with safe boundary margin."""
        w_img, h_img = pil_img.size
        x, y, w, h = box
        mx = int(w * margin)
        my = int(h * margin)
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(w_img, x + w + mx)
        y2 = min(h_img, y + h + my)
        return pil_img.crop((x1, y1, x2, y2))
