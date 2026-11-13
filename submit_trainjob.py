#!/usr/bin/env python3
"""Create one Trainer v2 TrainJob and wait for it to finish."""

import argparse
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

NAMESPACE = "cv-shard-training"
RUNTIME = "torch-distributed-cuda130-torch211-py312"
API = "https://kubernetes.default.svc"
TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
CA_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
SUCCESS = {"Complete", "Succeeded"}
FAILURE = {"Failed"}
TIMEOUT_SEC = 3600


def k8s(method, path, body=None, timeout=60):
    token = Path(TOKEN_PATH).read_text(encoding="utf-8").strip()
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(API + path, data=data, method=method)
    request.add_header("Authorization", f"Bearer {token}")
    if body is not None:
        request.add_header("Content-Type", "application/json")
    context = ssl.create_default_context(cafile=CA_PATH)
    try:
        with urllib.request.urlopen(request, context=context, timeout=timeout) as response:
            payload = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise SystemExit(f"{method} {path} failed: {error.code} {detail}") from error
    if not payload:
        return {}
    text = payload.decode("utf-8")
    if text[:1] in "{[":
        return json.loads(text)
    return text


def build_train_job(name, manifest_path, phase, metrics_dir, checkpoint, epochs, suspend=False):
    env = [
        {"name": "PYTHONPATH", "value": "/data/pydeps"},
        {"name": "METRICS_DIR", "value": metrics_dir},
        {"name": "CHECKPOINT", "value": checkpoint},
    ]
    if epochs:
        env.append({"name": "EPOCHS", "value": str(epochs)})
    return {
        "apiVersion": "trainer.kubeflow.org/v1alpha1",
        "kind": "TrainJob",
        "metadata": {"name": name, "namespace": NAMESPACE},
        "spec": {
            "suspend": suspend,
            "runtimeRef": {
                "name": RUNTIME,
                "kind": "ClusterTrainingRuntime",
                "apiGroup": "trainer.kubeflow.org",
            },
            "trainer": {
                "numNodes": 2,
                "numProcPerNode": 1,
                "resourcesPerNode": {
                    "requests": {"cpu": "4", "memory": "24Gi", "nvidia.com/gpu": "1"},
                    "limits": {"cpu": "4", "memory": "24Gi", "nvidia.com/gpu": "1"},
                },
                "env": env,
                "command": ["bash", "/data/code/torchrun.sh", manifest_path, phase],
            },
            "podTemplateOverrides": [
                {
                    "targetJobs": [{"name": "node"}],
                    "spec": {
                        "tolerations": [
                            {
                                "key": "nvidia.com/gpu",
                                "operator": "Exists",
                                "effect": "NoSchedule",
                            }
                        ],
                        "volumes": [
                            {
                                "name": "data",
                                "persistentVolumeClaim": {"claimName": "cv-data"},
                            },
                            {
                                "name": "dshm",
                                "emptyDir": {"medium": "Memory", "sizeLimit": "2Gi"},
                            },
                        ],
                        "containers": [
                            {
                                "name": "node",
                                "volumeMounts": [
                                    {"name": "data", "mountPath": "/data"},
                                    {"name": "dshm", "mountPath": "/dev/shm"},
                                ],
                            }
                        ],
                    },
                }
            ],
        },
    }


def condition_map(job):
    found = {}
    for condition in job.get("status", {}).get("conditions", []) or []:
        found[condition.get("type", "")] = condition
    return found


def matching_pods(name):
    listing = k8s("GET", f"/api/v1/namespaces/{NAMESPACE}/pods")
    pods = []
    for pod in listing.get("items", []):
        if name in pod["metadata"]["name"]:
            pods.append(pod)
    return pods


def describe_pods(name):
    lines = []
    for pod in matching_pods(name):
        phase = pod.get("status", {}).get("phase", "")
        node = pod.get("spec", {}).get("nodeName", "")
        waiting = []
        for container in pod.get("status", {}).get("containerStatuses", []) or []:
            state = container.get("state", {})
            if "waiting" in state:
                waiting.append(
                    f"{container['name']} {state['waiting'].get('reason', '')} {state['waiting'].get('message', '')}"
                )
        lines.append(f"{pod['metadata']['name']} phase={phase} node={node} {' '.join(waiting)}".rstrip())
    return lines


def wait_for_job(name, timeout_sec):
    deadline = time.time() + timeout_sec
    path = f"/apis/trainer.kubeflow.org/v1alpha1/namespaces/{NAMESPACE}/trainjobs/{name}"
    while time.time() < deadline:
        job = k8s("GET", path)
        conditions = condition_map(job)
        summary = ", ".join(
            f"{kind}={item.get('status')} {item.get('reason', '')}"
            for kind, item in conditions.items()
        )
        print(f"trainjob {name}: {summary or 'no conditions yet'}", flush=True)
        for line in describe_pods(name):
            print(line, flush=True)
        for kind in FAILURE:
            item = conditions.get(kind)
            if item and item.get("status") == "True":
                raise SystemExit(f"TrainJob failed: {item.get('message', '')}")
        for kind in SUCCESS:
            item = conditions.get(kind)
            if item and item.get("status") == "True":
                return job
        time.sleep(20)
    raise SystemExit(f"TrainJob {name} did not finish within {timeout_sec}s")


def gpu_metric_batch(metrics_dir, phase):
    rows = []
    pattern = re.compile(rf"gpu-{re.escape(phase)}-node-(\d+)")
    for path in sorted(Path(metrics_dir).glob(f"gpu-{phase}-node-*.jsonl")):
        match = pattern.fullmatch(path.stem)
        if match is None:
            continue
        node = f"node_{match.group(1)}"
        step_by_gpu = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            sample = json.loads(line)
            gpu = int(sample["gpu"])
            step = step_by_gpu.get(gpu, 0)
            step_by_gpu[gpu] = step + 1
            prefix = f"system/{node}/gpu_{gpu}_"
            stamp = int(sample["timestamp"])
            rows.append(
                (prefix + "utilization_percentage", float(sample["utilization_percentage"]), stamp, step)
            )
            rows.append(
                (prefix + "memory_usage_megabytes", float(sample["memory_usage_megabytes"]), stamp, step)
            )
    return rows


def log_gpu_metrics(metrics_dir, phase):
    run_id = os.environ.get("MLFLOW_RUN_ID", "")
    try:
        batch = gpu_metric_batch(metrics_dir, phase)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        print(f"gpu metrics unreadable: {error}", flush=True)
        return
    print(f"gpu_samples={len(batch)} mlflow_run_id_set={bool(run_id)}", flush=True)
    if not run_id or not batch:
        return
    try:
        import mlflow
        from mlflow.entities import Metric

        client = mlflow.MlflowClient()
        for start in range(0, len(batch), 200):
            metrics = [
                Metric(key, value, stamp, step)
                for key, value, stamp, step in batch[start : start + 200]
            ]
            client.log_batch(run_id=run_id, metrics=metrics)
    except Exception as error:
        print(f"gpu metrics logging failed: {error}", flush=True)


def tail_rank0(name):
    prefix = f"{name}-node-0"
    for pod in matching_pods(name):
        pod_name = pod["metadata"]["name"]
        if not pod_name.startswith(prefix):
            continue
        text = k8s(
            "GET",
            f"/api/v1/namespaces/{NAMESPACE}/pods/{pod_name}/log?container=node&tailLines=50",
        )
        print(f"--- log {pod_name} ---", flush=True)
        print(text if isinstance(text, str) else json.dumps(text), flush=True)
        return


def submit(manifest_path, metrics_path, timeout_sec, phase, metrics_dir, checkpoint, epochs):
    if phase not in {"train", "eval"}:
        raise SystemExit(f"unknown phase {phase}")
    source = Path(manifest_path)
    if source.is_dir():
        found = sorted(source.glob("*.json"))
        if not found:
            raise SystemExit(f"no manifest json in {source}")
        source = found[0]
    manifest = json.loads(source.read_text(encoding="utf-8"))
    job_manifest = manifest.get("manifest_path") or "/data/shards/manifest.json"
    if not Path(job_manifest).is_file():
        raise SystemExit(f"missing manifest on the shared volume: {job_manifest}")
    prefix = "cv-eval-" if phase == "eval" else "cv-train-"
    filename = "test-metrics.json" if phase == "eval" else "train-metrics.json"
    produced = f"{metrics_dir.rstrip('/')}/{filename}"
    name = prefix + time.strftime("%m%d%H%M%S")
    body = build_train_job(name, job_manifest, phase, metrics_dir, checkpoint, epochs)
    print(f"creating TrainJob {name} phase={phase} manifest={job_manifest}", flush=True)
    k8s(
        "POST",
        f"/apis/trainer.kubeflow.org/v1alpha1/namespaces/{NAMESPACE}/trainjobs",
        body,
    )
    wait_for_job(name, timeout_sec)
    try:
        tail_rank0(name)
    except SystemExit as error:
        print(f"pod log unavailable: {error}", flush=True)
    metrics_file = Path(produced)
    if not metrics_file.is_file():
        raise SystemExit(f"TrainJob finished without {produced}")
    target = Path(metrics_path)
    if target.exists() and target.is_dir():
        target = target / Path(produced).name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(metrics_file.read_text(encoding="utf-8"), encoding="utf-8")
    print(metrics_file.read_text(encoding="utf-8"), flush=True)
    print(f"trainjob={name}", flush=True)
    log_gpu_metrics(metrics_dir, phase)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--phase", choices=("train", "eval"), default="train")
    parser.add_argument("--timeout", type=int, default=TIMEOUT_SEC)
    parser.add_argument("--metrics-dir", default="/data/metrics")
    parser.add_argument("--checkpoint", default="/data/models/resnet18.pt")
    parser.add_argument("--epochs", type=int, default=0)
    args = parser.parse_args()
    submit(
        args.manifest,
        args.metrics,
        args.timeout,
        args.phase,
        args.metrics_dir,
        args.checkpoint,
        args.epochs,
    )


if __name__ == "__main__":
    main()
