#!/usr/bin/env python3
"""Pack a stratified split into train and test WebDataset shards."""

import argparse
import io
import json
import tarfile
from pathlib import Path

MIN_SHARDS = 2


def load_split(path):
    target = Path(path)
    if target.is_dir():
        found = sorted(target.glob("*.json"))
        if not found:
            raise SystemExit(f"no split json in {target}")
        target = found[0]
    payload = json.loads(target.read_text(encoding="utf-8"))
    for key in ("classes", "train", "test", "test_fraction", "split_seed"):
        if key not in payload:
            raise SystemExit(f"split file missing {key}")
    return payload


def write_shard(path, batch):
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as tar:
        for key, class_id, image_path in batch:
            tar.add(image_path, arcname=f"{key}.jpg")
            payload = f"{class_id}\n".encode()
            info = tarfile.TarInfo(f"{key}.cls")
            info.size = len(payload)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(payload))


def per_shard_size(n_images, images_per_shard):
    if n_images < MIN_SHARDS:
        raise SystemExit(f"need at least {MIN_SHARDS} images to build {MIN_SHARDS} shards, found {n_images}")
    return max(1, min(images_per_shard, n_images // MIN_SHARDS))


def write_split(samples, destination, images_per_shard):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for old in destination.glob("shard-*.tar"):
        old.unlink()
    per_shard = per_shard_size(len(samples), images_per_shard)
    shard_names = []
    total_bytes = 0
    for shard_index, start in enumerate(range(0, len(samples), per_shard)):
        chunk = samples[start : start + per_shard]
        batch = []
        for offset, (class_id, image_path) in enumerate(chunk):
            batch.append((f"{start + offset:06d}", class_id, image_path))
        name = f"shard-{shard_index:06d}.tar"
        path = destination / name
        write_shard(path, batch)
        size = path.stat().st_size
        total_bytes += size
        shard_names.append(name)
        print(f"wrote {destination.name}/{name} images={len(batch)} bytes={size}", flush=True)
    if len(shard_names) < MIN_SHARDS:
        raise SystemExit(f"{destination} has {len(shard_names)} shards; each trainer node needs one")
    return {
        "shard_dir": str(destination),
        "image_count": len(samples),
        "shard_count": len(shard_names),
        "shards": shard_names,
        "images_per_shard": per_shard,
        "bytes": total_bytes,
    }


def count_classes(samples, classes):
    totals = [0] * len(classes)
    for class_id, _path in samples:
        totals[class_id] += 1
    return {name: totals[index] for index, name in enumerate(classes)}


def samples_from(records):
    samples = []
    for record in records:
        samples.append((int(record["class_id"]), Path(record["path"])))
    if not samples:
        raise SystemExit("split section is empty")
    return samples


def pack(split_path, shard_dir, manifest_path, images_per_shard=2000):
    split = load_split(split_path)
    classes = split["classes"]
    train_samples = samples_from(split["train"])
    test_samples = samples_from(split["test"])
    destination = Path(shard_dir)
    destination.mkdir(parents=True, exist_ok=True)
    train = write_split(train_samples, destination / "train", images_per_shard)
    test = write_split(test_samples, destination / "test", images_per_shard)
    train_counts = count_classes(train_samples, classes)
    test_counts = count_classes(test_samples, classes)
    pvc_manifest = destination / "manifest.json"
    manifest = {
        "raw_dir": split.get("raw_dir", ""),
        "manifest_path": str(pvc_manifest),
        "classes": classes,
        "num_classes": len(classes),
        "test_fraction": split["test_fraction"],
        "split_seed": split["split_seed"],
        "image_count": len(train_samples) + len(test_samples),
        "bytes": train["bytes"] + test["bytes"],
        "images_per_shard": images_per_shard,
        "class_counts": {
            name: {"train": train_counts[name], "test": test_counts[name]} for name in classes
        },
        "train": train,
        "test": test,
    }
    encoded = json.dumps(manifest, indent=2) + "\n"
    pvc_manifest.write_text(encoded, encoding="utf-8")
    target = Path(manifest_path)
    if target.exists() and target.is_dir():
        target = target / "manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(encoded, encoding="utf-8")
    print(
        f"image_count={manifest['image_count']} train_shards={train['shard_count']} "
        f"test_shards={test['shard_count']} bytes={manifest['bytes']}",
        flush=True,
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--shards", default="/data/shards")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--images-per-shard", type=int, default=2000)
    args = parser.parse_args()
    pack(args.split, args.shards, args.manifest, args.images_per_shard)


if __name__ == "__main__":
    main()
