#!/usr/bin/env python3
"""Write labeled 256x256 JPEGs for the shard-training demo."""

import argparse
import os
import random
from pathlib import Path

from PIL import Image, ImageDraw

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


def draw_shape(class_id, rng):
    background = tuple(rng.randint(15, 70) for _ in range(3))
    image = Image.new("RGB", (256, 256), background)
    pen = ImageDraw.Draw(image)
    color = tuple(rng.randint(140, 255) for _ in range(3))
    shift = rng.randint(-12, 12)
    box = [36 + shift, 36 - shift, 220 + shift, 220 - shift]
    if class_id == 0:
        pen.ellipse(box, fill=color)
    elif class_id == 1:
        pen.rectangle(box, fill=color)
    elif class_id == 2:
        pen.polygon([(128, 32), (40, 216), (216, 216)], fill=color)
    elif class_id == 3:
        pen.polygon([(128, 28), (224, 128), (128, 228), (32, 128)], fill=color)
    elif class_id == 4:
        pen.rectangle([108, 28, 148, 228], fill=color)
        pen.rectangle([28, 108, 228, 148], fill=color)
    elif class_id == 5:
        pen.ellipse(box, outline=color, width=18)
    elif class_id == 6:
        pen.rectangle([28, 96, 228, 160], fill=color)
    elif class_id == 7:
        pen.rectangle([96, 28, 160, 228], fill=color)
    elif class_id == 8:
        for y in range(0, 256, 32):
            pen.rectangle([0, y, 256, y + 16], fill=color)
    else:
        for x in range(40, 220, 36):
            for y in range(40, 220, 36):
                pen.ellipse([x, y, x + 16, y + 16], fill=color)
    return image


def count_images(output):
    return sum(1 for _ in Path(output).glob("*/*.jpg"))


def sample_paths(output):
    root = Path(output)
    chosen = []
    for name in CLASSES:
        found = sorted((root / name).glob("*.jpg"))
        if found:
            chosen.append(found[0])
    return chosen


def generate(count, output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "classes.txt").write_text("\n".join(CLASSES) + "\n")
    per_class, extra = divmod(count, len(CLASSES))
    if count_images(out) == count:
        print(f"raw images already present: {count}", flush=True)
        return count

    written = 0
    for class_id, name in enumerate(CLASSES):
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        n_images = per_class + (1 if class_id < extra else 0)
        for index in range(n_images):
            path = folder / f"{index:06d}.jpg"
            if path.is_file() and path.stat().st_size > 0:
                continue
            rng = random.Random(class_id * 1_000_003 + index)
            draw_shape(class_id, rng).save(path, format="JPEG", quality=85)
            written += 1
            if written % 500 == 0:
                print(f"wrote {written}", flush=True)
    total = count_images(out)
    print(f"image_count={total} new_files={written}", flush=True)
    if total < count:
        raise SystemExit(f"expected {count} images, found {total}")
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20000)
    parser.add_argument(
        "--output",
        default=os.environ.get("DATA_ROOT", "/opt/app-root/src/data") + "/raw",
    )
    args = parser.parse_args()
    generate(args.count, args.output)


if __name__ == "__main__":
    main()
