#!/usr/bin/env python3
"""Train ResNet18 with DDP on WebDataset shards."""

import argparse
import glob
import json
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import webdataset as wds
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import models, transforms


def build_loader(shard_dir, batch_size, workers):
    shards = sorted(glob.glob(os.path.join(shard_dir, "shard-*.tar")))
    if not shards:
        raise SystemExit(f"no shards in {shard_dir}")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    def prepare(sample):
        image, label = sample
        text = label.decode() if isinstance(label, bytes) else str(label)
        return transform(image), int(text)

    dataset = (
        wds.WebDataset(
            shards,
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
    ), shards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--metrics", default="/data/metrics/train-metrics.json")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=20))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    loader, shards = build_loader(args.shards, args.batch_size, args.workers)
    if rank == 0:
        print(f"shards={len(shards)} world_size={dist.get_world_size()}", flush=True)

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model = DDP(model.to(device), device_ids=[local_rank])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    started = time.perf_counter()
    first_batch = None
    images = 0
    correct = 0
    total_loss = 0.0
    steps = 0
    for _epoch in range(args.epochs):
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

    if first_batch is None:
        first_batch = time.perf_counter() - started
    elapsed = time.perf_counter() - started
    reduced = torch.tensor(
        [images, correct, steps, total_loss], device=device, dtype=torch.float64
    )
    dist.all_reduce(reduced)
    if rank == 0:
        seen = float(reduced[0].item())
        metrics = {
            "images": int(seen),
            "steps": int(reduced[2].item()),
            "epoch_seconds": round(elapsed, 3),
            "images_per_sec": round(seen / elapsed, 2) if elapsed else 0,
            "time_to_first_batch_sec": round(first_batch, 3),
            "loss": round(float(reduced[3].item()) / max(int(reduced[2].item()), 1), 4),
            "train_accuracy": round(float(reduced[1].item()) / seen, 4) if seen else 0,
            "world_size": dist.get_world_size(),
            "batch_size": args.batch_size,
            "num_workers": args.workers,
            "shard_count": len(shards),
        }
        os.makedirs(os.path.dirname(args.metrics), exist_ok=True)
        with open(args.metrics, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps(metrics), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
