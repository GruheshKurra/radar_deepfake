import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(img_size: int = 224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.3),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=0.3),
        A.CoarseDropout(max_holes=4, max_height=16, max_width=16, p=0.2),
        A.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = 224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
        ToTensorV2(),
    ])


class DeepfakeDataset(Dataset):
    def __init__(
        self,
        image_paths: List[Path],
        labels: List[int],
        transform,
        preprocess_dir: Optional[Path] = None,
        validate_cache: bool = True,
    ):
        self.image_paths = image_paths
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform
        self.preprocess_dir = Path(preprocess_dir) if preprocess_dir else None

        if self.preprocess_dir and validate_cache:
            self._validate_preprocessed_features()

    def _validate_preprocessed_features(self):
        missing_count = 0
        total_count = len(self.image_paths)

        for idx, img_path in enumerate(self.image_paths):
            sobel_path = self._get_cache_path(img_path, "sobel")

            if sobel_path is None or not sobel_path.exists():
                missing_count += 1

    def _get_cache_path(self, img_path: Path, feature_type: str) -> Optional[Path]:
        if self.preprocess_dir is None:
            return None

        import hashlib
        try:
            with open(img_path, 'rb') as f:
                img_hash = hashlib.md5(f.read()).hexdigest()
        except Exception:
            img_hash = hashlib.md5(str(img_path).encode()).hexdigest()

        path_str = str(img_path)
        class_name = "real" if "/real/" in path_str or "\\real\\" in path_str else "fake"

        domain_name = "unknown"
        for domain in ["stylegan", "cifake", "wilddeepfake", "ff_c23"]:
            if domain in path_str:
                domain_name = domain
                break

        cache_dir = self.preprocess_dir / domain_name / class_name
        return cache_dir / f"{img_hash}_{feature_type}.npy"

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Union[
        Tuple[torch.Tensor, int, Dict[str, torch.Tensor]],
        Tuple[torch.Tensor, int]
    ]:
        try:
            img = Image.open(self.image_paths[idx]).convert("RGB")
            img_array = np.array(img)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load image at index {idx}: {self.image_paths[idx]}. "
                f"Error: {str(e)}"
            )

        img = self.transform(image=img_array)["image"] if self.transform else torch.from_numpy(img_array).permute(2, 0, 1)

        if not self.preprocess_dir:
            return img, self.labels[idx]

        sobel_path = self._get_cache_path(self.image_paths[idx], "sobel")
        if sobel_path is None or not sobel_path.exists():
            return img, self.labels[idx]

        extras = {"sobel_cached": torch.from_numpy(np.load(str(sobel_path)))}
        return img, self.labels[idx], extras
