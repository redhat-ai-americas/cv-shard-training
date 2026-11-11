#!/usr/bin/env python3
"""Pack labeled JPEGs into WebDataset tar shards."""

import argparse
import io
import json
import tarfile
from pathlib import Path

CLASSES = [
    "circle",
    "square",
    "triangle",
    "diamond",
    "cross",
    "ring",
    "hbar",
    "vbar",
    "stripes",
    "dots",
]


def collect_samples(raw_dir):
    samples = []
    root = Path(raw_dir)
    for class_id, name in enumerate(CLASSES):
        folder = root / name
        if not folder.is_dir():
            raise SystemExit(f"missing class folder: {folder}")
        files = sorted(folder.glob("*.jpg"))
        if not files:
            raise SystemExit(f"no JPEGs in {folder}")
        samples.extend((class_id, path) for path in files)
    return samples


def write_shard(path, batch):
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as tar:
        for key, class_id, image_path in batch:
            tar.add(image_path, arcname=f"{key}.jpg")
            payload = f"{class_id}\n".encode()
            info = tarfile.TarInfo(f"{key}.cls")
            info.size = len(payload)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(payload))


def pack(raw_dir, shard_dir, manifest_path, images_per_shard=2000):
    samples = collect_samples(raw_dir)
    if len(samples) < 2:
        raise SystemExit("need at least 2 images to build shards")
    per_shard = min(images_per_shard, max(1, len(samples) // 2))
    destination = Path(shard_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for old in destination.glob("shard-*.tar"):
        old.unlink()

    shard_names = []
    total_bytes = 0
    for shard_index, start in enumerate(range(0, len(samples), per_shard)):
        batch = []
        for offset, (class_id, image_path) in enumerate(samples[start : start + per_shard]):
            key = f"{start + offset:06d}"
            batch.append((key, class_id, image_path))
        name = f"shard-{shard_index:06d}.tar"
        path = destination / name
        write_shard(path, batch)
        total_bytes += path.stat().st_size
        shard_names.append(name)
        print(f"wrote {name} images={len(batch)} bytes={path.stat().st_size}", flush=True)

    if len(shard_names) < 2:
        raise SystemExit("expected at least 2 shards so each trainer node receives data")

    manifest = {
        "raw_dir": str(raw_dir),
        "shard_dir": str(destination),
        "image_count": len(samples),
        "shard_count": len(shard_names),
        "bytes": total_bytes,
        "images_per_shard": per_shard,
        "classes": CLASSES,
        "shards": shard_names,
    }
    encoded = json.dumps(manifest, indent=2) + "\n"
    target = Path(manifest_path)
    if target.exists() and target.is_dir():
        target = target / "manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(encoded)
    (destination / "manifest.json").write_text(encoded)
    print(
        f"image_count={manifest['image_count']} shard_count={manifest['shard_count']} bytes={total_bytes}",
        flush=True,
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="/data/raw")
    parser.add_argument("--shards", default="/data/shards")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--images-per-shard", type=int, default=2000)
    args = parser.parse_args()
    pack(args.raw, args.shards, args.manifest, args.images_per_shard)


if __name__ == "__main__":
    main()
