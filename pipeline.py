"""Split, pack, train, and evaluate as separate pipeline steps."""

import argparse
import json
import time

import kfp_server_api
from kfp import compiler, dsl, kubernetes
from kfp.client import Client
from kfp.dsl import Artifact, Input, Metrics, Output

BASE_IMAGE = (
    "image-registry.openshift-image-registry.svc:5000/"
    "redhat-ods-applications/s2i-generic-data-science-notebook:3.5"
)
PYPI = ["https://pypi.org/simple"]
PIPELINE_HOST = "https://ds-pipeline-dspa.cv-shard-training.svc.cluster.local:8443"
DASHBOARD = "https://rh-ai.apps.ocp.9j696.sandbox5421.opentlc.com"
PACKAGE = "/opt/app-root/src/cv_shard_pipeline.yaml"
NAMESPACE = "cv-shard-training"
PIPELINE_NAME = "overhead-shard-training"
VERSION_NAME = "v2"
EXPERIMENT_NAME = "overhead-shard-training"
MLFLOW_EXPERIMENT = "overhead-shard-training"
DESCRIPTION = "Split, pack, train, and evaluate the overhead image classifier"


@dsl.component(base_image=BASE_IMAGE, pip_index_urls=PYPI)
def split_dataset(
    raw_dir: str,
    shard_dir: str,
    max_per_class: int,
    split_file: Output[Artifact],
    metrics: Output[Metrics],
):
    import json
    import subprocess
    import sys
    from pathlib import Path

    shared = str(Path(shard_dir).parent / "splits" / "split.json")
    subprocess.check_call(
        [
            sys.executable,
            "/data/code/split_dataset.py",
            "--raw",
            raw_dir,
            "--output",
            split_file.path,
            "--max-per-class",
            str(max_per_class),
            "--shared",
            shared,
        ]
    )
    target = Path(split_file.path)
    if target.is_dir():
        target = target / "split.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    metrics.log_metric("train_count", float(payload["train_count"]))
    metrics.log_metric("test_count", float(payload["test_count"]))


@dsl.component(base_image=BASE_IMAGE, pip_index_urls=PYPI)
def pack_shards(
    shard_dir: str,
    split_file: Input[Artifact],
    manifest: Output[Artifact],
    metrics: Output[Metrics],
):
    import json
    import subprocess
    import sys
    from pathlib import Path

    subprocess.check_call(
        [
            sys.executable,
            "/data/code/pack_shards.py",
            "--split",
            split_file.path,
            "--shards",
            shard_dir,
            "--manifest",
            manifest.path,
        ]
    )
    target = Path(manifest.path)
    if target.is_dir():
        target = target / "manifest.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    metrics.log_metric("image_count", float(payload["image_count"]))
    metrics.log_metric("train_shards", float(payload["train"]["shard_count"]))
    metrics.log_metric("test_shards", float(payload["test"]["shard_count"]))


@dsl.component(base_image=BASE_IMAGE, pip_index_urls=PYPI)
def train(
    metrics_dir: str,
    checkpoint: str,
    epochs: int,
    manifest: Input[Artifact],
    report: Output[Artifact],
    metrics: Output[Metrics],
):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    subprocess.check_call(
        [
            sys.executable,
            "/data/code/submit_trainjob.py",
            "--manifest",
            manifest.path,
            "--metrics",
            report.path,
            "--phase",
            "train",
            "--metrics-dir",
            metrics_dir,
            "--checkpoint",
            checkpoint,
            "--epochs",
            str(epochs),
        ]
    )
    def class_key(label):
        return "".join(ch if ch.isalnum() else "_" for ch in str(label))

    target = Path(report.path)
    if target.is_dir():
        target = target / "train-metrics.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    metrics.log_metric("images_per_sec", float(payload["images_per_sec"]))
    run_id = os.environ.get("MLFLOW_RUN_ID", "")
    history = payload.get("history") or []
    mlflow_env = sorted(key for key in os.environ if key.startswith("MLFLOW_"))
    print(
        f"mlflow_run_id={run_id} history_points={len(history)} mlflow_env={','.join(mlflow_env)}",
        flush=True,
    )
    logged_curve = False
    if run_id and history:
        try:
            import mlflow

            for point in history:
                step = int(point["epoch"])
                mlflow.log_metric("loss", float(point["loss"]), step=step)
                mlflow.log_metric("train_accuracy", float(point["accuracy"]), step=step)
                mlflow.log_metric("train_precision_macro", float(point["precision_macro"]), step=step)
                mlflow.log_metric("train_recall_macro", float(point["recall_macro"]), step=step)
                mlflow.log_metric("train_f1_macro", float(point["f1_macro"]), step=step)
                for row in point["per_class"]:
                    name = class_key(row["class"])
                    mlflow.log_metric(f"train_precision_{name}", float(row["precision"]), step=step)
                    mlflow.log_metric(f"train_recall_{name}", float(row["recall"]), step=step)
                    mlflow.log_metric(f"train_f1_{name}", float(row["f1"]), step=step)
            logged_curve = True
        except Exception as error:
            print(f"mlflow history logging failed: {error}", flush=True)
    if not logged_curve:
        metrics.log_metric("loss", float(payload["loss"]))
        metrics.log_metric("train_accuracy", float(payload["train_accuracy"]))
        metrics.log_metric("train_precision_macro", float(payload["precision_macro"]))
        metrics.log_metric("train_recall_macro", float(payload["recall_macro"]))
        metrics.log_metric("train_f1_macro", float(payload["f1_macro"]))
        for row in payload["per_class"]:
            name = class_key(row["class"])
            metrics.log_metric(f"train_precision_{name}", float(row["precision"]))
            metrics.log_metric(f"train_recall_{name}", float(row["recall"]))
            metrics.log_metric(f"train_f1_{name}", float(row["f1"]))


@dsl.component(base_image=BASE_IMAGE, pip_index_urls=PYPI)
def evaluate(
    metrics_dir: str,
    checkpoint: str,
    manifest: Input[Artifact],
    report: Output[Artifact],
    metrics: Output[Metrics],
):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    def class_key(label):
        return "".join(ch if ch.isalnum() else "_" for ch in str(label))

    subprocess.check_call(
        [
            sys.executable,
            "/data/code/submit_trainjob.py",
            "--manifest",
            manifest.path,
            "--metrics",
            report.path,
            "--phase",
            "eval",
            "--metrics-dir",
            metrics_dir,
            "--checkpoint",
            checkpoint,
        ]
    )
    target = Path(report.path)
    if target.is_dir():
        target = target / "test-metrics.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    metrics.log_metric("test_loss", float(payload["test_loss"]))
    metrics.log_metric("test_accuracy", float(payload["test_accuracy"]))
    metrics.log_metric("test_images", float(payload["test_images"]))
    metrics.log_metric("test_precision_macro", float(payload["precision_macro"]))
    metrics.log_metric("test_recall_macro", float(payload["recall_macro"]))
    metrics.log_metric("test_f1_macro", float(payload["f1_macro"]))
    metrics.log_metric("test_precision_weighted", float(payload["precision_weighted"]))
    metrics.log_metric("test_recall_weighted", float(payload["recall_weighted"]))
    metrics.log_metric("test_f1_weighted", float(payload["f1_weighted"]))
    for row in payload["per_class"]:
        name = class_key(row["class"])
        metrics.log_metric(f"test_precision_{name}", float(row["precision"]))
        metrics.log_metric(f"test_recall_{name}", float(row["recall"]))
        metrics.log_metric(f"test_f1_{name}", float(row["f1"]))
        metrics.log_metric(f"test_support_{name}", float(row["support"]))
    classes = payload["classes"]
    matrix = payload["confusion_matrix"]
    for true_index, true_name in enumerate(classes):
        for pred_index, pred_name in enumerate(classes):
            metrics.log_metric(
                f"test_cm_{class_key(true_name)}_{class_key(pred_name)}",
                float(matrix[true_index][pred_index]),
            )
    run_id = os.environ.get("MLFLOW_RUN_ID", "")
    mlflow_env = sorted(key for key in os.environ if key.startswith("MLFLOW_"))
    print(f"mlflow_run_id={run_id} mlflow_env={','.join(mlflow_env)}", flush=True)
    if run_id:
        try:
            import mlflow

            lines = ["true\\predicted\t" + "\t".join(classes)]
            for index, name in enumerate(classes):
                counts = "\t".join(str(int(value)) for value in matrix[index])
                lines.append(f"{name}\t{counts}")
            mlflow.log_text("\n".join(lines) + "\n", "confusion_matrix.txt")
            mlflow.log_dict(
                {
                    "classes": classes,
                    "matrix": matrix,
                    "per_class": payload["per_class"],
                },
                "confusion_matrix.json",
            )
        except Exception as error:
            print(f"mlflow confusion matrix file failed: {error}", flush=True)
        try:
            import mlflow
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            figure, axis = plt.subplots(figsize=(6.2, 5.2))
            values = [[float(item) for item in row] for row in matrix]
            image = axis.imshow(values, cmap="Blues")
            axis.set_xticks(range(len(classes)))
            axis.set_xticklabels(classes, rotation=30, ha="right")
            axis.set_yticks(range(len(classes)))
            axis.set_yticklabels(classes)
            axis.set_xlabel("Predicted")
            axis.set_ylabel("True")
            axis.set_title("Test confusion matrix")
            peak = max(max(row) for row in values) if values else 0
            for row_index, row in enumerate(values):
                for col_index, value in enumerate(row):
                    color = "white" if peak and value > peak / 2 else "black"
                    axis.text(
                        col_index,
                        row_index,
                        str(int(value)),
                        ha="center",
                        va="center",
                        color=color,
                    )
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            figure.tight_layout()
            mlflow.log_figure(figure, "confusion_matrix.png")
            plt.close(figure)
        except Exception as error:
            print(f"mlflow confusion matrix figure failed: {error}", flush=True)


def attach(task, cpu, memory_request, memory_limit):
    task.set_caching_options(False)
    task.set_cpu_request(cpu).set_cpu_limit("1")
    task.set_memory_request(memory_request).set_memory_limit(memory_limit)
    kubernetes.mount_pvc(task, pvc_name="cv-data", mount_path="/data")
    return task


@dsl.pipeline(name=PIPELINE_NAME, description=DESCRIPTION)
def cv_shard_pipeline(
    raw_dir: str = "/data/raw",
    shard_dir: str = "/data/shards",
    metrics_dir: str = "/data/metrics",
    checkpoint: str = "/data/models/resnet18.pt",
    max_per_class: int = 0,
    epochs: int = 1,
):
    split = attach(
        split_dataset(raw_dir=raw_dir, shard_dir=shard_dir, max_per_class=max_per_class),
        "500m",
        "512Mi",
        "2Gi",
    )
    packed = attach(
        pack_shards(shard_dir=shard_dir, split_file=split.outputs["split_file"]),
        "500m",
        "1Gi",
        "2Gi",
    )
    trained = attach(
        train(
            metrics_dir=metrics_dir,
            checkpoint=checkpoint,
            epochs=epochs,
            manifest=packed.outputs["manifest"],
        ),
        "500m",
        "512Mi",
        "1Gi",
    )
    scored = attach(
        evaluate(
            metrics_dir=metrics_dir,
            checkpoint=checkpoint,
            manifest=packed.outputs["manifest"],
        ),
        "500m",
        "512Mi",
        "1Gi",
    )
    scored.after(trained)


def compile_pipeline(path=PACKAGE):
    compiler.Compiler().compile(cv_shard_pipeline, path)
    return path


def pipeline_id_named(client, name):
    query = json.dumps(
        {
            "predicates": [
                {
                    "operation": "EQUALS",
                    "key": "display_name",
                    "stringValue": name,
                }
            ]
        }
    )
    found = client.list_pipelines(namespace=NAMESPACE, page_size=5, filter=query)
    matches = found.pipelines or []
    if len(matches) > 1:
        raise SystemExit(f"multiple pipelines named {name}")
    if not matches:
        return None
    return matches[0].pipeline_id


def version_id_named(client, pipeline_id, name):
    token = ""
    while True:
        page = client.list_pipeline_versions(pipeline_id, page_token=token, page_size=50)
        for version in page.pipeline_versions or []:
            if version.display_name == name:
                return version.pipeline_version_id
        token = page.next_page_token or ""
        if not token:
            return None


def register(client, path):
    pipeline_id = pipeline_id_named(client, PIPELINE_NAME)
    if pipeline_id is None:
        created = client.upload_pipeline(
            pipeline_package_path=path,
            pipeline_name=PIPELINE_NAME,
            description=DESCRIPTION,
            namespace=NAMESPACE,
        )
        pipeline_id = created.pipeline_id
    version_id = version_id_named(client, pipeline_id, VERSION_NAME)
    if version_id is None:
        uploaded = client.upload_pipeline_version(
            pipeline_package_path=path,
            pipeline_version_name=VERSION_NAME,
            pipeline_id=pipeline_id,
            description=DESCRIPTION,
        )
        version_id = uploaded.pipeline_version_id
    return pipeline_id, version_id


def submit(path=PACKAGE, parameters=None):
    compile_pipeline(path)
    token = open("/var/run/secrets/kubernetes.io/serviceaccount/token", encoding="utf-8").read()
    client = Client(
        host=PIPELINE_HOST,
        existing_token=token,
        namespace=NAMESPACE,
        verify_ssl=False,
    )
    pipeline_id, version_id = register(client, path)
    try:
        experiment = client.get_experiment(experiment_name=EXPERIMENT_NAME, namespace=NAMESPACE)
    except Exception:
        experiment = client.create_experiment(
            name=EXPERIMENT_NAME,
            description=DESCRIPTION,
            namespace=NAMESPACE,
        )
    run_name = "overhead-" + time.strftime("%H%M%S")
    reference = kfp_server_api.V2beta1PipelineVersionReference(
        pipeline_id=pipeline_id,
        pipeline_version_id=version_id,
    )
    run_kwargs = {
        "display_name": run_name,
        "description": DESCRIPTION,
        "experiment_id": experiment.experiment_id,
        "pipeline_version_reference": reference,
        "plugins_input": {"mlflow": {"experiment_name": MLFLOW_EXPERIMENT}},
    }
    if parameters:
        run_kwargs["runtime_config"] = kfp_server_api.V2beta1RuntimeConfig(parameters=parameters)
    run_body = kfp_server_api.V2beta1Run(**run_kwargs)
    run = client._run_api.run_service_create_run(run=run_body)
    print(f"pipeline_name={PIPELINE_NAME}")
    print(f"pipeline_id={pipeline_id}")
    print(f"pipeline_version={VERSION_NAME}")
    print(f"pipeline_version_id={version_id}")
    print(f"kfp_experiment={EXPERIMENT_NAME}")
    print(f"mlflow_experiment={MLFLOW_EXPERIMENT}")
    print(f"run_name={run_name}")
    print(f"run_id={run.run_id}")
    print(f"plugins_output={run.plugins_output}")
    print(
        f"Open {DASHBOARD} and go to CV shard training, Develop and train, Pipelines. "
        f"The pipeline {PIPELINE_NAME}, version {VERSION_NAME}, owns this run. "
        f"MLflow experiment {MLFLOW_EXPERIMENT} tracks it."
    )
    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-per-class", type=int, default=0)
    parser.add_argument("--raw-dir", default="/data/raw")
    parser.add_argument("--shard-dir", default="/data/shards")
    parser.add_argument("--metrics-dir", default="/data/metrics")
    parser.add_argument("--checkpoint", default="/data/models/resnet18.pt")
    args = parser.parse_args()
    changed = (
        args.epochs != 1
        or args.max_per_class != 0
        or args.raw_dir != "/data/raw"
        or args.shard_dir != "/data/shards"
        or args.metrics_dir != "/data/metrics"
        or args.checkpoint != "/data/models/resnet18.pt"
    )
    if args.max_per_class > 0 and args.shard_dir.rstrip("/") == "/data/shards":
        raise SystemExit(
            "A subset run deletes shards in --shard-dir. Use a directory other than /data/shards."
        )
    if args.max_per_class > 0 and args.checkpoint == "/data/models/resnet18.pt":
        raise SystemExit("A subset run must use a --checkpoint other than /data/models/resnet18.pt.")
    if args.max_per_class > 0 and args.metrics_dir.rstrip("/") == "/data/metrics":
        raise SystemExit("A subset run must use a --metrics-dir other than /data/metrics.")
    parameters = None
    if changed:
        parameters = {
            "raw_dir": args.raw_dir,
            "shard_dir": args.shard_dir,
            "metrics_dir": args.metrics_dir,
            "checkpoint": args.checkpoint,
            "max_per_class": args.max_per_class,
            "epochs": args.epochs,
        }
    submit(parameters=parameters)


if __name__ == "__main__":
    main()
