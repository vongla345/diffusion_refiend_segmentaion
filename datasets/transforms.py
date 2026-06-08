"""Thin wrappers over `online_aug` compose pipelines."""

from datasets.online_aug import (
    MEAN,
    STD,
    compose_train_view_a,
    compose_train_view_b,
    compose_val,
)


def train_tf(patch_size: int = 256):
    return compose_train_view_a(patch_size)


def stain_tf(patch_size: int = 256):
    return compose_train_view_b(patch_size)


def val_tf(patch_size: int = 256):
    return compose_val(patch_size)


__all__ = ["MEAN", "STD", "train_tf", "stain_tf", "val_tf"]
