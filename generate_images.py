#!/usr/bin/env python3
"""Write labeled 256x256 overhead JPEGs for the shard-training demo."""

import argparse
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CLASSES = ["truck", "boat", "plane", "building"]
STAMP = "overhead-silhouettes-v1"
CANVAS = 256
SPRITE = 200
OUTLINE = (32, 36, 40, 255)
PALETTES = (
    ((92, 86, 62), (138, 128, 96)),
    ((70, 78, 58), (120, 124, 96)),
    ((110, 104, 96), (168, 162, 150)),
    ((84, 78, 70), (130, 118, 102)),
    ((64, 72, 68), (108, 114, 108)),
)
RESAMPLE = getattr(Image, "Resampling", Image).BICUBIC
BILINEAR = getattr(Image, "Resampling", Image).BILINEAR


def box(n, x0, y0, x1, y1):
    return [
        int(round(x0 * n)),
        int(round(y0 * n)),
        int(round(x1 * n)),
        int(round(y1 * n)),
    ]


def points(n, pairs):
    return [(int(round(x * n)), int(round(y * n))) for x, y in pairs]


def tint(color, factor):
    return tuple(max(0, min(255, int(channel * factor))) for channel in color[:3]) + (255,)


def rounded(pen, rect, radius, fill, outline=None, width=1):
    left, top, right, bottom = rect
    span = min(abs(right - left), abs(bottom - top))
    radius = max(0, min(radius, span // 2 - 1))
    if radius >= 2 and hasattr(pen, "rounded_rectangle"):
        pen.rounded_rectangle(rect, radius=radius, fill=fill, outline=outline, width=width)
    else:
        pen.rectangle(rect, fill=fill, outline=outline, width=width)


def terrain(rng):
    field = np.random.default_rng(rng.randrange(2**31)).random((32, 32), dtype=np.float32)
    image = Image.fromarray((field * 255).astype(np.uint8), mode="L").resize((CANVAS, CANVAS), BILINEAR)
    tone = np.asarray(image).astype(np.float32) / 255.0
    dark, light = PALETTES[rng.randrange(len(PALETTES))]

    def jitter(color):
        return np.array(
            [min(255, max(0, channel + rng.randint(-12, 12))) for channel in color],
            dtype=np.float32,
        )

    blended = jitter(dark) * (1.0 - tone[..., None]) + jitter(light) * tone[..., None]
    return Image.fromarray(blended.astype(np.uint8), mode="RGB")


def draw_truck(pen, n, rng, color):
    length = rng.uniform(0.72, 0.90)
    bed_w = rng.uniform(0.30, 0.44)
    cab_frac = rng.uniform(0.24, 0.36)
    cab_w = bed_w * rng.uniform(0.66, 0.84)
    x0 = 0.5 - length / 2
    x1 = x0 + length
    cab_x = x0 + length * cab_frac
    bed = box(n, x0, 0.5 - bed_w / 2, x1, 0.5 + bed_w / 2)
    cab = box(n, x0, 0.5 - cab_w / 2, cab_x, 0.5 + cab_w / 2)
    rounded(pen, bed, max(2, n // 30), color, OUTLINE, 2)
    rounded(pen, cab, max(2, n // 22), tint(color, 0.76), OUTLINE, 2)
    window = box(
        n,
        cab_x - length * 0.08,
        0.5 - cab_w * 0.28,
        cab_x - length * 0.015,
        0.5 + cab_w * 0.28,
    )
    pen.rectangle(window, fill=tint(color, 0.42))
    wheel_r = bed_w * rng.uniform(0.16, 0.22)
    axles = [x0 + length * 0.18, x0 + length * rng.uniform(0.70, 0.84)]
    if rng.random() < 0.45:
        axles.insert(1, x0 + length * rng.uniform(0.46, 0.58))
    for axle in axles:
        for edge in (0.5 - bed_w / 2, 0.5 + bed_w / 2):
            pen.ellipse(
                box(n, axle - wheel_r, edge - wheel_r, axle + wheel_r, edge + wheel_r),
                fill=(22, 22, 24, 255),
                outline=tint(color, 0.35),
                width=2,
            )


def draw_boat(pen, n, rng, color):
    length = rng.uniform(0.70, 0.90)
    beam = rng.uniform(0.20, 0.34)
    stern = 0.5 - length / 2
    bow = stern + length
    flare = beam * rng.uniform(0.08, 0.16)
    hull = [
        (stern, 0.5 - beam / 2 + flare),
        (stern + length * 0.20, 0.5 - beam / 2),
        (bow - length * 0.16, 0.5 - beam * 0.22),
        (bow, 0.5),
        (bow - length * 0.16, 0.5 + beam * 0.22),
        (stern + length * 0.20, 0.5 + beam / 2),
        (stern, 0.5 + beam / 2 - flare),
    ]
    pen.polygon(points(n, hull), fill=color, outline=OUTLINE)
    house_l = length * rng.uniform(0.16, 0.28)
    house_w = beam * rng.uniform(0.40, 0.62)
    house_x = stern + length * rng.uniform(0.18, 0.34)
    rounded(
        pen,
        box(n, house_x, 0.5 - house_w / 2, house_x + house_l, 0.5 + house_w / 2),
        max(2, n // 40),
        tint(color, 1.18),
        OUTLINE,
        2,
    )


def draw_plane(pen, n, rng, color):
    length = rng.uniform(0.60, 0.80)
    fuse_w = rng.uniform(0.08, 0.13)
    span = rng.uniform(0.72, 0.94)
    chord = rng.uniform(0.12, 0.18)
    sweep = rng.uniform(0.04, 0.14)
    tail_x = 0.5 - length / 2
    nose_x = tail_x + length
    root_x = 0.5 + length * rng.uniform(-0.02, 0.08)
    half = span / 2
    wing = [
        (root_x + chord * 0.55, 0.5 - fuse_w * 0.2),
        (root_x + chord * 0.10 - sweep, 0.5 - half),
        (root_x - chord * 0.40 - sweep * 0.45, 0.5 - half),
        (root_x - chord * 0.45, 0.5 - fuse_w * 0.2),
        (root_x - chord * 0.45, 0.5 + fuse_w * 0.2),
        (root_x - chord * 0.40 - sweep * 0.45, 0.5 + half),
        (root_x + chord * 0.10 - sweep, 0.5 + half),
        (root_x + chord * 0.55, 0.5 + fuse_w * 0.2),
    ]
    pen.polygon(points(n, wing), fill=color)
    fuse = [
        (tail_x, 0.5 - fuse_w * 0.28),
        (0.5, 0.5 - fuse_w * 0.55),
        (nose_x - length * 0.08, 0.5 - fuse_w * 0.20),
        (nose_x, 0.5),
        (nose_x - length * 0.08, 0.5 + fuse_w * 0.20),
        (0.5, 0.5 + fuse_w * 0.55),
        (tail_x, 0.5 + fuse_w * 0.28),
    ]
    pen.polygon(points(n, fuse), fill=color, outline=OUTLINE)
    tail_span = span * rng.uniform(0.22, 0.34)
    tail_chord = chord * rng.uniform(0.42, 0.68)
    tx = tail_x + length * 0.07
    tail = [
        (tx + tail_chord * 0.35, 0.5 - fuse_w * 0.15),
        (tx - tail_chord * 0.15, 0.5 - tail_span / 2),
        (tx - tail_chord * 0.55, 0.5 - tail_span / 2),
        (tx - tail_chord * 0.20, 0.5 - fuse_w * 0.15),
        (tx - tail_chord * 0.20, 0.5 + fuse_w * 0.15),
        (tx - tail_chord * 0.55, 0.5 + tail_span / 2),
        (tx - tail_chord * 0.15, 0.5 + tail_span / 2),
        (tx + tail_chord * 0.35, 0.5 + fuse_w * 0.15),
    ]
    pen.polygon(points(n, tail), fill=tint(color, 0.86), outline=OUTLINE)


def draw_building(pen, n, rng, color):
    kind = rng.randrange(3)
    holes = []
    blocks = []
    if kind == 0:
        wide = rng.uniform(0.46, 0.78)
        tall = rng.uniform(0.46, 0.78)
        if max(wide, tall) > min(wide, tall) * 1.65:
            short = max(wide, tall) / rng.uniform(1.05, 1.45)
            if wide >= tall:
                tall = short
            else:
                wide = short
        blocks.append((0.5 - wide / 2, 0.5 - tall / 2, 0.5 + wide / 2, 0.5 + tall / 2))
    elif kind == 1:
        margin = 0.10
        outer = 1.0 - 2 * margin
        thickness = outer * rng.uniform(0.40, 0.58)
        blocks.append((margin, margin, margin + outer, margin + thickness))
        blocks.append((margin, margin, margin + thickness, margin + outer))
    else:
        wide = rng.uniform(0.55, 0.82)
        tall = rng.uniform(0.55, 0.82)
        x0, y0 = 0.5 - wide / 2, 0.5 - tall / 2
        blocks.append((x0, y0, x0 + wide, y0 + tall))
        inset = rng.uniform(0.24, 0.34)
        holes.append(
            box(
                n,
                x0 + wide * inset,
                y0 + tall * inset,
                x0 + wide * (1 - inset),
                y0 + tall * (1 - inset),
            )
        )
    for index, (x0, y0, x1, y1) in enumerate(blocks):
        fill = color if index == 0 else tint(color, 0.84)
        pen.rectangle(box(n, x0, y0, x1, y1), fill=fill, outline=OUTLINE, width=2)
        rows = rng.randint(2, 4)
        cols = rng.randint(2, 4)
        pad_x = (x1 - x0) * 0.14
        pad_y = (y1 - y0) * 0.14
        cell_w = ((x1 - x0) - 2 * pad_x) / cols
        cell_h = ((y1 - y0) - 2 * pad_y) / rows
        for row in range(rows):
            for col in range(cols):
                wx0 = x0 + pad_x + col * cell_w + cell_w * 0.18
                wy0 = y0 + pad_y + row * cell_h + cell_h * 0.18
                pen.rectangle(
                    box(n, wx0, wy0, wx0 + cell_w * 0.55, wy0 + cell_h * 0.55),
                    fill=tint(color, 0.55),
                )
    for _ in range(rng.randint(1, 3)):
        host = blocks[0]
        rw = (host[2] - host[0]) * rng.uniform(0.08, 0.16)
        rh = (host[3] - host[1]) * rng.uniform(0.08, 0.16)
        rx = rng.uniform(host[0] + 0.08 * (host[2] - host[0]), host[2] - rw - 0.04)
        ry = rng.uniform(host[1] + 0.08 * (host[3] - host[1]), host[3] - rh - 0.04)
        pen.rectangle(box(n, rx, ry, rx + rw, ry + rh), fill=tint(color, 1.2), outline=OUTLINE)
    return holes


def punch(sprite, holes):
    if not holes:
        return
    alpha = sprite.getchannel("A")
    mask = ImageDraw.Draw(alpha)
    for hole in holes:
        mask.rectangle(hole, fill=0)
    sprite.putalpha(alpha)


def draw_sprite(class_id, rng, color):
    sprite = Image.new("RGBA", (SPRITE, SPRITE), (0, 0, 0, 0))
    pen = ImageDraw.Draw(sprite)
    holes = []
    if class_id == 0:
        draw_truck(pen, SPRITE, rng, color)
    elif class_id == 1:
        draw_boat(pen, SPRITE, rng, color)
    elif class_id == 2:
        draw_plane(pen, SPRITE, rng, color)
    else:
        holes = draw_building(pen, SPRITE, rng, color)
    punch(sprite, holes)
    return sprite


def paste_position(rng, width, height, visible=0.75):
    def axis(span):
        low = int(round(span * (visible - 1)))
        high = int(round(CANVAS - visible * span))
        if low > high:
            center = (CANVAS - span) // 2
            return center, center
        return low, high

    x0, x1 = axis(width)
    y0, y1 = axis(height)
    return rng.randint(x0, x1), rng.randint(y0, y1)


def render(class_id, seed):
    rng = random.Random(seed)
    image = terrain(rng)
    color = tuple(rng.randint(80, 235) for _ in range(3)) + (255,)
    sprite = draw_sprite(class_id, rng, color)
    scale = rng.uniform(0.50, 1.15)
    size = max(24, int(round(sprite.width * scale)))
    sprite = sprite.resize((size, size), RESAMPLE)
    sprite = sprite.rotate(
        rng.uniform(0, 360),
        resample=RESAMPLE,
        expand=True,
        fillcolor=(0, 0, 0, 0),
    )
    image.paste(sprite, paste_position(rng, sprite.width, sprite.height), sprite)
    return image


def count_images(output):
    return sum(1 for _ in Path(output).glob("*/*.jpg"))


def sample_grid(output, per_class=4):
    root = Path(output)
    chosen = []
    for name in CLASSES:
        found = sorted(
            (path for path in (root / name).glob("*.jpg") if path.stem.isdigit()),
            key=lambda path: int(path.stem),
        )
        if not found:
            continue
        take = min(per_class, len(found))
        if take == 1:
            chosen.append(found[0])
            continue
        for slot in range(take):
            index = int(round(slot * (len(found) - 1) / (take - 1)))
            chosen.append(found[index])
    return chosen


def class_lines(path):
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sync_folder(folder, n_images):
    folder.mkdir(parents=True, exist_ok=True)
    present = set()
    for path in folder.glob("*.jpg"):
        if not path.stem.isdigit():
            path.unlink()
            continue
        index = int(path.stem)
        if index >= n_images or path.stat().st_size == 0:
            path.unlink()
            continue
        present.add(index)
    return [index for index in range(n_images) if index not in present]


def _write_one(task):
    class_id, index, path = task
    render(class_id, class_id * 1_000_003 + index).save(path, format="JPEG", quality=85)


def generate(count, output, workers=1, replace=False):
    if count < 1:
        raise SystemExit("count must be at least 1")
    if workers < 1:
        raise SystemExit("workers must be at least 1")
    out = Path(output)
    existing = count_images(out) if out.exists() else 0
    if existing > count and not replace:
        raise SystemExit(
            f"{out} already has {existing} images. Refusing to shrink it to {count}. "
            "Pass replace=True, or --replace on the command line, to delete the extras."
        )
    out.mkdir(parents=True, exist_ok=True)
    stamp_path = out / "generator.txt"
    classes_path = out / "classes.txt"
    stamp_ok = stamp_path.is_file() and stamp_path.read_text(encoding="utf-8").strip() == STAMP
    if not stamp_ok or class_lines(classes_path) != CLASSES:
        for child in list(out.iterdir()):
            if child.is_dir():
                shutil.rmtree(child)
    stamp_path.write_text(STAMP + "\n", encoding="utf-8")
    classes_path.write_text("\n".join(CLASSES) + "\n", encoding="utf-8")

    per_class, extra = divmod(count, len(CLASSES))
    tasks = []
    for class_id, name in enumerate(CLASSES):
        n_images = per_class + (1 if class_id < extra else 0)
        for index in sync_folder(out / name, n_images):
            tasks.append((class_id, index, str(out / name / f"{index:06d}.jpg")))
    if not tasks and count_images(out) == count:
        print(f"raw images already present: {count}", flush=True)
        return count

    if workers == 1:
        for done, task in enumerate(tasks, start=1):
            _write_one(task)
            if done % 500 == 0:
                print(f"wrote {done}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for done, _ in enumerate(pool.map(_write_one, tasks, chunksize=32), start=1):
                if done % 500 == 0:
                    print(f"wrote {done}", flush=True)
    total = count_images(out)
    print(f"image_count={total} new_files={len(tasks)}", flush=True)
    if total != count:
        raise SystemExit(f"expected {count} images, found {total}")
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete existing images when the folder already has more than --count",
    )
    parser.add_argument(
        "--output",
        default=os.environ.get("DATA_ROOT", "/opt/app-root/src/data") + "/raw",
    )
    args = parser.parse_args()
    generate(args.count, args.output, args.workers, replace=args.replace)


if __name__ == "__main__":
    main()
