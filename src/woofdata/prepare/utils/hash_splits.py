"""Assign train / val / test from sha1 hashes, the same on every machine."""

from __future__ import annotations

import hashlib

HASH_BUCKETS: int = 100  # split_from_hash: sha1(key) mod 100


def split_from_hash(key: str, train_below: int, val_below: int) -> str:
    """Return train / val / test from sha1(key) mod 100, the same on every machine.

    A bucket below train_below is train, below val_below is val, the rest is test.
    """
    bucket = int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16) % HASH_BUCKETS
    if bucket < train_below:
        return "train"
    if bucket < val_below:
        return "val"
    return "test"


def balanced_group_splits(
    images_per_group: dict[str, int],
    test_share: float,
    val_share: float,
    priority_groups: set[str] | None = None,
    min_priority_groups: int = 0,
) -> dict[str, str]:
    """Return {group: split} with about test_share / val_share of the images in test / val.

    Groups are walked in sha1(group name) order, the same on every machine. test is filled
    first, then val: each first takes min_priority_groups of the priority_groups (e.g. whole
    recordings, so an evaluation split is not just one or two scenes), then continues the
    walk until it holds its share of all images. Every group left goes to train. A group is
    never divided; a split ends up with fewer priority groups only if none are left.
    """
    if priority_groups is None:
        priority_groups = set()
    total_images = sum(images_per_group.values())
    hashed_groups: list[tuple[str, str]] = []
    for group in images_per_group:
        hashed_groups.append((hashlib.sha1(group.encode("utf-8")).hexdigest(), group))
    hashed_groups.sort()

    split_by_group: dict[str, str] = {}
    for split, share in (("test", test_share), ("val", val_share)):
        filled = 0
        priority_taken = 0
        for _, group in hashed_groups:
            if priority_taken >= min_priority_groups:
                break
            if group in priority_groups and group not in split_by_group:
                split_by_group[group] = split
                filled += images_per_group[group]
                priority_taken += 1
        for _, group in hashed_groups:
            if filled >= share * total_images:
                break
            if group not in split_by_group:
                split_by_group[group] = split
                filled += images_per_group[group]
    for _, group in hashed_groups:
        split_by_group.setdefault(group, "train")
    return split_by_group
