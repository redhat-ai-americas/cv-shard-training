#!/usr/bin/env python3
"""Append GPU utilization samples until this process is killed.

The training pod writes the file. It does not call MLflow. submit_trainjob.py
reads the file after the TrainJob finishes and logs the series on the pipeline
task's nested run, under the system/ names MLflow groups as System metrics.
"""

import json
import os
import subprocess
import time
from pathlib import Path

INTERVAL_SEC = 15


def sample_path():
    directory = Path(os.environ.get("METRICS_DIR", "/data/metrics"))
    phase = os.environ.get("GPU_SAMPLE_PHASE", "train")
    rank = os.environ.get("PET_NODE_RANK", "0")
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"gpu-{phase}-node-{rank}.jsonl"


def parse_line(line, timestamp):
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 3:
        return None
    index, util, mem_mib = parts
    if util in {"", "N/A", "[N/A]"} or mem_mib in {"", "N/A", "[N/A]"}:
        return None
    return {
        "timestamp": timestamp,
        "gpu": int(index),
        "utilization_percentage": float(util),
        "memory_usage_megabytes": round(float(mem_mib) * 1024 * 1024 / 1e6, 1),
    }


def query(timestamp):
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    rows = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        row = parse_line(line, timestamp)
        if row is not None:
            rows.append(row)
    return rows


def main():
    path = sample_path()
    path.write_text("", encoding="utf-8")
    while True:
        stamp = int(time.time() * 1000)
        try:
            rows = query(stamp)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            print(f"gpu sample skipped: {error}", flush=True)
            time.sleep(INTERVAL_SEC)
            continue
        if rows:
            with path.open("a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
                handle.flush()
        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main()
