#!/usr/bin/env python3
"""Train or evaluate ResNet18 with DDP on WebDataset shards."""

import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import webdataset as wds
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import models, transforms


def load_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("classes", "num_classes", "train", "test"):
        if key not in manifest:
            raise SystemExit(f"manifest missing {key}")
    if int(manifest["num_classes"]) != len(manifest["classes"]):
        raise SystemExit("num_classes does not match classes")
    return manifest


def shard_paths(split):
    root = split["shard_dir"]
    paths = [os.path.join(root, name) for name in split["shards"]]
    missing = [item for item in paths if not os.path.isfile(item)]
    if not paths or missing:
        found = missing[0] if missing else root
        raise SystemExit(f"missing shard {found}")
    return paths


def build_loader(paths, batch_size, workers):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    def prepare(sample):
        image, label = sample
        text = label.decode() if isinstance(label, bytes) else str(label)
        return transform(image), int(text.strip())

    dataset = (
        wds.WebDataset(
            paths,
            shardshuffle=False,
            empty_check=False,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
        )
        .decode("pil")
        .to_tuple("jpg", "cls")
        .map(prepare)
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
    )


CHECKPOINT = os.environ.get("CHECKPOINT", "/data/models/resnet18.pt")
LEARNING_RATE = 0.05
EPOCHS = 1


def write_metrics(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(payload), flush=True)


def save_checkpoint(model, classes):
    target = Path(CHECKPOINT)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.module.state_dict(), "classes": classes}, target)


def load_model(num_classes, device):
    target = Path(CHECKPOINT)
    if not target.is_file():
        raise SystemExit(f"missing checkpoint {target}")
    try:
        blob = torch.load(target, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(target, map_location=device)
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(blob["state_dict"])
    return model.to(device)


def reduce_stats(device, images, correct, steps, total_loss):
    reduced = torch.tensor(
        [images, correct, steps, total_loss], device=device, dtype=torch.float64
    )
    dist.all_reduce(reduced)
    return reduced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--phase", choices=("train", "eval"), default="train")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--metrics", default="")
    args = parser.parse_args()
    if args.epochs < 1:
        raise SystemExit("epochs must be at least 1")
    if not args.metrics:
        args.metrics = (
            "/data/metrics/test-metrics.json" if args.phase == "eval" else "/data/metrics/train-metrics.json"
        )

    manifest = load_manifest(args.manifest)
    num_classes = int(manifest["num_classes"])
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=20))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    loss_fn = torch.nn.CrossEntropyLoss()

    if args.phase == "train":
        shards = shard_paths(manifest["train"])
        if rank == 0:
            print(
                f"phase=train classes={','.join(manifest['classes'])} "
                f"shards={len(shards)} world_size={dist.get_world_size()}",
                flush=True,
            )
        loader = build_loader(shards, args.batch_size, args.workers)
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model = DDP(model.to(device), device_ids=[local_rank])
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
        started = time.perf_counter()
        first_batch = None
        total_images = 0.0
        last_reduced = None
        history = []
        for epoch in range(args.epochs):
            images = 0
            correct = 0
            total_loss = 0.0
            steps = 0
            for images_batch, labels in loader:
                if first_batch is None:
                    first_batch = time.perf_counter() - started
                    if rank == 0:
                        print(f"time_to_first_batch_sec={first_batch:.3f}", flush=True)
                images_batch = images_batch.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images_batch)
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()
                images += images_batch.size(0)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total_loss += float(loss.item())
                steps += 1
            last_reduced = reduce_stats(device, images, correct, steps, total_loss)
            total_images += float(last_reduced[0].item())
            if rank == 0:
                seen = float(last_reduced[0].item())
                epoch_steps = max(int(last_reduced[2].item()), 1)
                epoch_loss = float(last_reduced[3].item()) / epoch_steps
                epoch_accuracy = float(last_reduced[1].item()) / seen if seen else 0
                print(
                    f"epoch={epoch + 1}/{args.epochs} loss={epoch_loss:.4f} "
                    f"accuracy={epoch_accuracy:.4f}",
                    flush=True,
                )
                history.append(
                    {
                        "epoch": epoch + 1,
                        "loss": round(epoch_loss, 4),
                        "accuracy": round(epoch_accuracy, 4),
                    }
                )
        if first_batch is None:
            first_batch = time.perf_counter() - started
        elapsed = time.perf_counter() - started
        del loader
        if rank == 0:
            save_checkpoint(model, manifest["classes"])
            seen = float(last_reduced[0].item())
            train_steps = max(int(last_reduced[2].item()), 1)
            write_metrics(
                args.metrics,
                {
                    "phase": "train",
                    "model": "resnet18",
                    "images": int(seen),
                    "steps": int(last_reduced[2].item()),
                    "epoch_seconds": round(elapsed, 3),
                    "images_per_sec": round(total_images / elapsed, 2) if elapsed else 0,
                    "time_to_first_batch_sec": round(first_batch, 3),
                    "loss": round(float(last_reduced[3].item()) / train_steps, 4),
                    "train_accuracy": round(float(last_reduced[1].item()) / seen, 4) if seen else 0,
                    "history": history,
                    "classes": manifest["classes"],
                    "num_classes": num_classes,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": LEARNING_RATE,
                    "world_size": dist.get_world_size(),
                    "num_workers": args.workers,
                    "shard_count": len(shards),
                    "checkpoint": CHECKPOINT,
                },
            )
    else:
        shards = shard_paths(manifest["test"])
        if rank == 0:
            print(
                f"phase=eval classes={','.join(manifest['classes'])} "
                f"shards={len(shards)} world_size={dist.get_world_size()}",
                flush=True,
            )
        model = load_model(num_classes, device)
        model.eval()
        loader = build_loader(shards, args.batch_size, args.workers)
        images = 0
        correct = 0
        total_loss = 0.0
        steps = 0
        started = time.perf_counter()
        with torch.no_grad():
            for images_batch, labels in loader:
                images_batch = images_batch.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images_batch)
                loss = loss_fn(logits, labels)
                images += images_batch.size(0)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total_loss += float(loss.item())
                steps += 1
        elapsed = time.perf_counter() - started
        reduced = reduce_stats(device, images, correct, steps, total_loss)
        del loader
        if rank == 0:
            seen = float(reduced[0].item())
            scored_steps = max(int(reduced[2].item()), 1)
            write_metrics(
                args.metrics,
                {
                    "phase": "eval",
                    "model": "resnet18",
                    "test_images": int(seen),
                    "test_steps": int(reduced[2].item()),
                    "test_seconds": round(elapsed, 3),
                    "test_loss": round(float(reduced[3].item()) / scored_steps, 4),
                    "test_accuracy": round(float(reduced[1].item()) / seen, 4) if seen else 0,
                    "classes": manifest["classes"],
                    "num_classes": num_classes,
                    "checkpoint": CHECKPOINT,
                    "world_size": dist.get_world_size(),
                    "batch_size": args.batch_size,
                    "shard_count": len(shards),
                },
            )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
