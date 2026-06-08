import glob
import logging
import os
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from datasets.online_aug import compose_diffusion_image_normalize, compose_diffusion_spatial, compose_diffusion_val
from datasets.transforms import stain_tf, train_tf, val_tf

logger = logging.getLogger(__name__)

IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def list_image_files(img_dir: str) -> list[str]:
    paths = []
    for ext in sorted(IMG_EXTS):
        paths.extend(glob.glob(os.path.join(img_dir, f"*{ext}")))
    return sorted(paths, key=lambda p: Path(p).name)


def find_mask_for_image(mask_dir: Optional[str], image_path: str) -> Optional[str]:
    if not mask_dir:
        return None
    stem = Path(image_path).stem
    for ext in sorted(IMG_EXTS):
        candidate = os.path.join(mask_dir, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


def convert_and_copy(src_dir: str, dst_img_dir: str, dst_mask_dir: str) -> Tuple[int, int]:
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_mask_dir, exist_ok=True)
    n_img = n_mask = 0

    sub_dirs = {
        d.lower(): d
        for d in os.listdir(src_dir)
        if os.path.isdir(os.path.join(src_dir, d))
    }

    img_src = next(
        (os.path.join(src_dir, sub_dirs[k]) for k in ["image", "images", "img"] if k in sub_dirs),
        src_dir,
    )
    mask_src = next(
        (
            os.path.join(src_dir, sub_dirs[k])
            for k in ["mask", "masks", "annotation", "anno", "gt"]
            if k in sub_dirs
        ),
        None,
    )

    for fpath in sorted(glob.glob(f"{img_src}/*")):
        if Path(fpath).suffix.lower() not in IMG_EXTS:
            continue
        stem = Path(fpath).stem
        if any(stem.endswith(s) for s in ["_anno", "_mask", "_gt", "_label"]):
            continue
        img = cv2.imread(fpath)
        if img is None:
            continue
        cv2.imwrite(f"{dst_img_dir}/{stem}.png", img)
        n_img += 1

    if mask_src and os.path.exists(mask_src):
        for fpath in sorted(glob.glob(f"{mask_src}/*")):
            if Path(fpath).suffix.lower() not in IMG_EXTS:
                continue
            stem = Path(fpath).stem
            for suf in ["_anno", "_mask", "_gt", "_label"]:
                if stem.endswith(suf):
                    stem = stem[: -len(suf)]
            m = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            cv2.imwrite(f"{dst_mask_dir}/{stem}.png", (m > 0).astype(np.uint8) * 255)
            n_mask += 1
    else:
        for fpath in sorted(glob.glob(f"{img_src}/*")):
            if Path(fpath).suffix.lower() not in IMG_EXTS:
                continue
            stem = Path(fpath).stem
            for suf in ["_anno", "_mask", "_gt", "_label"]:
                if stem.endswith(suf):
                    img_stem = stem[: -len(suf)]
                    m = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                    if m is None:
                        continue
                    cv2.imwrite(f"{dst_mask_dir}/{img_stem}.png", (m > 0).astype(np.uint8) * 255)
                    n_mask += 1
                    break

    return n_img, n_mask


def find_crag_dataset_root(extract_tmp: str) -> str:
    dataset_root = None
    for root, dirs, files in os.walk(extract_tmp):
        try:
            contents = os.listdir(root)
        except OSError:
            continue
        if any(d in contents for d in ["test", "val"]) and any(
            "train" in c.lower() for c in contents
        ):
            dataset_root = root
            break
    if not dataset_root:
        raise FileNotFoundError(f"Dataset root not found under {extract_tmp}")
    return dataset_root


def _split_map_for_crag(dataset_root: str) -> Dict[str, str]:
    split_map = {}
    for d in os.listdir(dataset_root):
        dl = d.lower()
        if "train_sup" in dl or dl == "labeled":
            split_map[d] = "labeled"
        elif "train_unsup" in dl or "unlabeled" in dl:
            split_map[d] = "unlabeled"
        elif "val" in dl:
            split_map[d] = "val"
        elif "test" in dl:
            split_map[d] = "test"
    return split_map


def resolve_split_dirs(data_root: str) -> Dict[str, Tuple[str, Optional[str]]]:
    """
    Resolve split directories for both layouts:
    1) normalized: data_root/{labeled,unlabeled,val,test}/images|masks
    2) raw 10%:    data_root/{train_sup_x,train_unsup_x,val,test}/image|mask
    """
    normalized = {
        "labeled": (f"{data_root}/labeled/images", f"{data_root}/labeled/masks"),
        "unlabeled": (f"{data_root}/unlabeled/images", f"{data_root}/unlabeled/masks"),
        "val": (f"{data_root}/val/images", f"{data_root}/val/masks"),
        "test": (f"{data_root}/test/images", f"{data_root}/test/masks"),
    }
    if all(os.path.isdir(v[0]) for v in normalized.values()):
        return normalized

    split_map = _split_map_for_crag(data_root)
    if not split_map:
        raise FileNotFoundError(
            f"Cannot resolve dataset splits under {data_root}. "
            "Expected normalized labeled/unlabeled/val/test or raw train_sup/train_unsup/val/test."
        )

    resolved: Dict[str, Tuple[str, Optional[str]]] = {}
    for src_name, split in split_map.items():
        src_dir = os.path.join(data_root, src_name)
        sub_dirs = {
            d.lower(): d
            for d in os.listdir(src_dir)
            if os.path.isdir(os.path.join(src_dir, d))
        }
        img_dir = next(
            (os.path.join(src_dir, sub_dirs[k]) for k in ["image", "images", "img"] if k in sub_dirs),
            src_dir,
        )
        mask_dir = next(
            (
                os.path.join(src_dir, sub_dirs[k])
                for k in ["mask", "masks", "annotation", "anno", "gt"]
                if k in sub_dirs
            ),
            None,
        )
        if split == "unlabeled":
            resolved[split] = (img_dir, None)
        else:
            resolved[split] = (img_dir, mask_dir)
    return resolved


def prepare_crag_layout(
    zip_path: str,
    extract_tmp: str,
    data_root: str,
    force_extract: bool = False,
) -> None:
    if zip_path and (force_extract or not os.path.exists(extract_tmp)):
        logger.info("Extracting ZIP to %s", extract_tmp)
        os.makedirs(extract_tmp, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_tmp)
    elif not os.path.isdir(extract_tmp):
        if not zip_path:
            logger.info(
                "prepare_crag_layout: no zip_path and no %s - skip (use existing data_root).",
                extract_tmp,
            )
            return
        raise FileNotFoundError(
            f"No extract dir {extract_tmp}. Pass zip_path or extract manually."
        )

    dataset_root = find_crag_dataset_root(extract_tmp)
    logger.info("Dataset root: %s", dataset_root)
    split_map = _split_map_for_crag(dataset_root)
    logger.info("Split map: %s", split_map)

    for src_name, split in split_map.items():
        n_i, n_m = convert_and_copy(
            os.path.join(dataset_root, src_name),
            f"{data_root}/{split}/images",
            f"{data_root}/{split}/masks",
        )
        logger.info("  %s -> %s: %d images | %d masks", src_name, split, n_i, n_m)


def extract_patches(
    src_img_dir: str,
    src_mask_dir: Optional[str],
    dst_img_dir: str,
    dst_mask_dir: str,
    patch_size: int = 256,
    stride: int = 200,
    min_gland: float = 0.03,
) -> int:
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_mask_dir, exist_ok=True)
    img_paths = sorted(glob.glob(f"{src_img_dir}/*.png"))
    total = 0

    for img_path in tqdm(img_paths, desc=f"  {Path(src_img_dir).parent.name}"):
        stem = Path(img_path).stem
        img = cv2.imread(img_path)
        if img is None:
            continue

        mask_path = f"{src_mask_dir}/{stem}.png" if src_mask_dir else None
        mask = (
            cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_path and os.path.exists(mask_path)
            else None
        )
        if src_mask_dir and mask is None:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)

        h, w = img.shape[:2]
        ph = (patch_size - h % patch_size) % patch_size
        pw = (patch_size - w % patch_size) % patch_size
        img_p = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        mask_p = np.pad(mask, ((0, ph), (0, pw)), mode="constant") if mask is not None else None

        r = 0
        while r + patch_size <= img_p.shape[0]:
            c = 0
            while c + patch_size <= img_p.shape[1]:
                ip = img_p[r : r + patch_size, c : c + patch_size]
                mp = mask_p[r : r + patch_size, c : c + patch_size] if mask_p is not None else None

                gray = cv2.cvtColor(ip, cv2.COLOR_BGR2GRAY)
                if np.mean(gray < 230) < 0.15:
                    c += stride
                    continue
                if mp is not None and np.mean(mp > 0) < min_gland:
                    c += stride
                    continue

                name = f"{stem}_r{r:04d}_c{c:04d}.png"
                cv2.imwrite(f"{dst_img_dir}/{name}", ip)
                if mp is not None:
                    cv2.imwrite(f"{dst_mask_dir}/{name}", (mp > 0).astype(np.uint8) * 255)
                total += 1
                c += stride
            r += stride
    return total


def extract_all_splits_to_patches(
    data_root: str,
    patch_root: str,
    patch_size: int,
    stride: int,
    min_gland: float,
    clear_patch_root: bool = True,
) -> None:
    import shutil

    if clear_patch_root and os.path.exists(patch_root):
        shutil.rmtree(patch_root)
    os.makedirs(patch_root, exist_ok=True)

    resolved = resolve_split_dirs(data_root)
    for split in ["labeled", "unlabeled", "val", "test"]:
        if split not in resolved:
            logger.warning("Skip split %s (missing in %s)", split, data_root)
            continue
        src_img, src_mask = resolved[split]
        if not os.path.exists(src_img):
            logger.warning("Skip split %s (missing %s)", split, src_img)
            continue
        n = extract_patches(
            src_img,
            src_mask,
            f"{patch_root}/{split}/images",
            f"{patch_root}/{split}/masks",
            patch_size,
            stride,
            min_gland,
        )
        logger.info("  %s: %d patches", split, n)


class LabeledDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        is_train: bool = True,
        patch_size: int = 256,
        crop_retry: int = 1,
        min_fg_ratio: float = 0.0,
    ):
        self.paths = list_image_files(img_dir)
        self.mask_dir = mask_dir
        self.crop_retry = max(1, int(crop_retry))
        self.min_fg_ratio = float(min_fg_ratio)
        self.tf_a = train_tf(patch_size) if is_train else val_tf(patch_size)
        self.tf_b = stain_tf(patch_size)
        self.is_train = is_train
        if len(self.paths) == 0:
            raise ValueError(f"No images in {img_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        name = Path(self.paths[idx]).name
        img = cv2.cvtColor(cv2.imread(self.paths[idx]), cv2.COLOR_BGR2RGB)
        mask_path = find_mask_for_image(self.mask_dir, self.paths[idx])
        mask = (
            cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_path and os.path.exists(mask_path)
            else None
        )
        if mask is None:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
        n_try = self.crop_retry if self.is_train else 1
        aug_a = None
        for _attempt in range(n_try):
            aug_a = self.tf_a(image=img, mask=mask)
            msk_try = (aug_a["mask"] > 0).float()
            fg_ratio = msk_try.mean().item()
            if (
                self.min_fg_ratio <= 0
                or fg_ratio >= self.min_fg_ratio
                or not self.is_train
            ):
                break
        assert aug_a is not None
        img_a = aug_a["image"]
        msk = (aug_a["mask"] > 0).float().unsqueeze(0)
        img_b = self.tf_b(image=img)["image"] if self.is_train else img_a.clone()
        return {"image_a": img_a, "image_b": img_b, "mask": msk, "name": name}


class UnlabeledDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        patch_size: int = 256,
    ):
        self.paths = list_image_files(img_dir)
        self.tf_a = train_tf(patch_size)
        self.tf_b = stain_tf(patch_size)
        if len(self.paths) == 0:
            raise ValueError(f"No images in {img_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        name = Path(self.paths[idx]).name
        img = cv2.cvtColor(cv2.imread(self.paths[idx]), cv2.COLOR_BGR2RGB)
        return {
            "image_a": self.tf_a(image=img)["image"],
            "image_b": self.tf_b(image=img)["image"],
            "name": name,
        }


class DiffusionDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        patch_size: int = 256,
        crop_retry: int = 1,
        min_fg_ratio: float = 0.0,
        is_train: bool = True,
    ):
        self.img_paths = list_image_files(img_dir)
        self.mask_dir = mask_dir
        self.crop_retry = max(1, int(crop_retry))
        self.min_fg_ratio = float(min_fg_ratio)
        # Training: random crop + flips; Validation: deterministic resize only
        self.spatial = compose_diffusion_spatial(patch_size) if is_train else compose_diffusion_val(patch_size)
        self.img_norm = compose_diffusion_image_normalize()
        if len(self.img_paths) == 0:
            raise ValueError(f"No images in {img_dir}")

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx):
        name = Path(self.img_paths[idx]).name
        img = cv2.cvtColor(cv2.imread(self.img_paths[idx]), cv2.COLOR_BGR2RGB)
        mask_path = find_mask_for_image(self.mask_dir, self.img_paths[idx])
        mask = (
            cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_path and os.path.exists(mask_path)
            else None
        )
        if mask is None:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
        aug = None
        for _ in range(self.crop_retry):
            aug = self.spatial(image=img, mask=mask)
            m = np.asarray(aug["mask"], dtype=np.float32)
            fg_ratio = float((m > 127).sum()) / max(m.size, 1)
            if self.min_fg_ratio <= 0 or fg_ratio >= self.min_fg_ratio:
                break
        assert aug is not None
        img_t = self.img_norm(image=aug["image"])["image"]
        m = np.asarray(aug["mask"], dtype=np.float32)
        mask_t = torch.as_tensor(m / 127.5 - 1.0).unsqueeze(0)
        return {"image": img_t, "mask": mask_t, "name": name}
