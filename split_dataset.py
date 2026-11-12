#!/usr/bin/env python3
"""Stratified train/test split of the labeled JPEG corpus."""

import argparse
import json
import random
from pathlib import Path

SPLIT_SEED = 17
TEST_FRACTION = 0.2


def load_classes(raw_dir):
    path = Path(raw_dir) / "classes.txt"
    if not path.is_file():
        raise SystemExit(f"missing class list: {path}")
    classes = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(classes) < 2:
        raise SystemExit(f"need at least 2 classes in {path}")
    return classes


def image_paths(folder):
    paths = [path for path in folder.glob("*.jpg") if path.stem.isdigit()]
    return sorted(paths, key=lambda path: int(path.stem))


def collect_samples(raw_dir, classes):
    grouped = []
    root = Path(raw_dir)
    for class_id, name in enumerate(classes):
        folder = root / name
        if not folder.is_dir():
            raise SystemExit(f"missing class folder: {folder}")
        files = image_paths(folder)
        if len(files) < 2:
            raise SystemExit(f"need at least 2 JPEGs in {folder}")
        grouped.append([(class_id, path) for path in files])
    return grouped


def stratified_split(grouped, test_fraction, seed):
    rng = random.Random(seed)
    train, test = [], []
    for group in grouped:
        items = list(group)
        rng.shuffle(items)
        n_test = int(round(len(items) * test_fraction))
        n_test = min(max(n_test, 1), len(items) - 1)
        test.extend(items[:n_test])
        train.extend(items[n_test:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def count_classes(samples, classes):
    totals = [0] * len(classes)
    for class_id, _path in samples:
        totals[class_id] += 1
    return {name: totals[index] for index, name in enumerate(classes)}


def as_records(samples):
    return [{"class_id": class_id, "path": str(path)} for class_id, path in samples]


def artifact_file(path, filename):
    target = Path(path)
    if target.exists() and target.is_dir():
        target = target / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def write_split(raw_dir, output_path, test_fraction=TEST_FRACTION, split_seed=SPLIT_SEED):
    if not 0 < test_fraction < 1:
        raise SystemExit("test fraction must be between 0 and 1")
    classes = load_classes(raw_dir)
    train_samples, test_samples = stratified_split(
        collect_samples(raw_dir, classes), test_fraction, split_seed
    )
    train_counts = count_classes(train_samples, classes)
    test_counts = count_classes(test_samples, classes)
    payload = {
        "raw_dir": str(raw_dir),
        "classes": classes,
        "num_classes": len(classes),
        "test_fraction": test_fraction,
        "split_seed": split_seed,
        "train_count": len(train_samples),
        "test_count": len(test_samples),
        "class_counts": {
            name: {"train": train_counts[name], "test": test_counts[name]} for name in classes
        },
        "train": as_records(train_samples),
        "test": as_records(test_samples),
    }
    shared = Path("/data/splits/split.json")
    if str(raw_dir).startswith("/data"):
        payload["split_path"] = str(shared)
    encoded = json.dumps(payload) + "\n"
    if "split_path" in payload:
        shared.parent.mkdir(parents=True, exist_ok=True)
        shared.write_text(encoded, encoding="utf-8")
    artifact_file(output_path, "split.json").write_text(encoded, encoding="utf-8")
    print(
        f"train_count={payload['train_count']} test_count={payload['test_count']} "
        f"classes={','.join(classes)}",
        flush=True,
    )
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="/data/raw")
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()
    write_split(args.raw, args.output, args.test_fraction, args.split_seed)


if __name__ == "__main__":
    main()
