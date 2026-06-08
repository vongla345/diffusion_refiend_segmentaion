import cv2
from typing import Any, Dict, List, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _pad_reflect101(patch_size: int) -> A.PadIfNeeded:
    return A.PadIfNeeded(
        min_height=patch_size,
        min_width=patch_size,
        border_mode=cv2.BORDER_REFLECT_101,
    )


def train_view_a_spatial(patch_size: int = 256) -> List[Any]:
    return [
        _pad_reflect101(patch_size),
        A.RandomCrop(height=patch_size, width=patch_size, p=1.0),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
            translate_percent=(-0.05, 0.05),
            scale=(0.9, 1.1),
            rotate=(-30, 30),
            p=0.5,
        ),
    ]


def train_view_b_spatial(patch_size: int = 256) -> List[Any]:
    return [
        _pad_reflect101(patch_size),
        A.RandomCrop(height=patch_size, width=patch_size, p=1.0),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
            translate_percent=(-0.1, 0.1),
            scale=(0.8, 1.2),
            rotate=(-45, 45),
            shear=(-10, 10),
            p=0.5,
        ),
    ]


def stain_photometric_transforms() -> List[Any]:
    return [
        A.OneOf(
            [
                A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=1.0),
                A.HueSaturationValue(20, 30, 20, p=1.0),
                A.RandomBrightnessContrast(0.3, 0.3, p=1.0),
            ],
            p=0.8,
        ),
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                # Reduced from (0.1, 1.0) — original was larger than the image
                # signal itself, creating an irreducible prediction gap between
                # the strong and weak views.
                A.GaussNoise(std_range=(0.02, 0.3), p=1.0),
            ],
            p=0.4,
        ),
        A.CoarseDropout(
            # Reduced from [1, 8] — too many dropped patches prevented any
            # consistent segmentation signal from the strong view.
            num_holes_range=[1, 3],
            hole_height_range=[16, 32],
            hole_width_range=[16, 32],
            p=0.5,
        ),
    ]


def imagenet_normalize_transforms(
    mean: Tuple[float, ...] = MEAN,
    std: Tuple[float, ...] = STD,
) -> List[Any]:
    return [A.Normalize(mean=mean, std=std), ToTensorV2()]


def diffusion_spatial_list(patch_size: int = 256) -> List[Any]:
    return [
        _pad_reflect101(patch_size),
        A.RandomCrop(height=patch_size, width=patch_size, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]


def diffusion_image_normalize_transforms() -> List[Any]:
    return [
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ]


def compose_train_view_a(patch_size: int = 256) -> A.Compose:
    return A.Compose(train_view_a_spatial(patch_size) + imagenet_normalize_transforms())


def compose_train_view_b(patch_size: int = 256) -> A.Compose:
    return A.Compose(
        train_view_b_spatial(patch_size)
        + stain_photometric_transforms()
        + imagenet_normalize_transforms()
    )


def compose_val(patch_size: int = 256) -> A.Compose:
    return A.Compose([A.Resize(patch_size, patch_size, p=1)] + imagenet_normalize_transforms())


def compose_diffusion_spatial(patch_size: int = 256) -> A.Compose:
    return A.Compose(diffusion_spatial_list(patch_size))


def compose_diffusion_val(patch_size: int = 256) -> A.Compose:
    """Deterministic resize for diffusion validation (no random augmentation)."""
    return A.Compose([A.Resize(patch_size, patch_size, p=1)])


def compose_diffusion_image_normalize() -> A.Compose:
    return A.Compose(diffusion_image_normalize_transforms())


def data_normalize_2d(
    mean: Tuple[float, ...] = MEAN,
    std: Tuple[float, ...] = STD,
) -> A.Compose:
    return A.Compose(imagenet_normalize_transforms(mean, std))


def data_transform_2d(patch_size: int = 256) -> Dict[str, A.Compose]:
    """Spatial-only pipelines (no normalize) for train/val keys."""
    return {
        "train": A.Compose(train_view_a_spatial(patch_size)),
        "val": A.Compose([A.Resize(patch_size, patch_size, p=1)]),
        "test": A.Compose([A.Resize(patch_size, patch_size, p=1)]),
    }
