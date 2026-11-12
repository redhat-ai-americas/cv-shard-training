"""Split, pack, train, and evaluate as separate pipeline steps."""

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
VERSION_NAME = "v1"
EXPERIMENT_NAME = "overhead-shard-training"
MLFLOW_EXPERIMENT = "overhead-shard-training"
DESCRIPTION = "Split, pack, train, and evaluate the overhead image classifier"


@dsl.component(base_image=BASE_IMAGE, pip_index_urls=PYPI)
def split_dataset(split_file: Output[Artifact], metrics: Output[Metrics]):
    import json
    import subprocess
    import sys
    from pathlib import Path

    subprocess.check_call(
        [
            sys.executable,
            "/data/code/split_dataset.py",
            "--raw",
            "/data/raw",
            "--output",
            split_file.path,
        ]
    )
    target = Path(split_file.path)
    if target.is_dir():
        target = target / "split.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    metrics.log_metric("train_count", float(payload["train_count"]))
    metrics.log_metric("test_count", float(payload["test_count"]))


@dsl.component(base_image=BASE_IMAGE, pip_index_urls=PYPI)
def pack_shards(split_file: Input[Artifact], manifest: Output[Artifact], metrics: Output[Metrics]):
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
            "/data/shards",
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
def train(manifest: Input[Artifact], report: Output[Artifact], metrics: Output[Metrics]):
    import json
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
        ]
    )
    target = Path(report.path)
    if target.is_dir():
        target = target / "train-metrics.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    metrics.log_metric("loss", float(payload["loss"]))
    metrics.log_metric("train_accuracy", float(payload["train_accuracy"]))
    metrics.log_metric("images_per_sec", float(payload["images_per_sec"]))


@dsl.component(base_image=BASE_IMAGE, pip_index_urls=PYPI)
def evaluate(manifest: Input[Artifact], report: Output[Artifact], metrics: Output[Metrics]):
    import json
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
            "eval",
        ]
    )
    target = Path(report.path)
    if target.is_dir():
        target = target / "test-metrics.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    metrics.log_metric("test_loss", float(payload["test_loss"]))
    metrics.log_metric("test_accuracy", float(payload["test_accuracy"]))
    metrics.log_metric("test_images", float(payload["test_images"]))


def attach(task, cpu, memory_request, memory_limit):
    task.set_caching_options(False)
    task.set_cpu_request(cpu).set_cpu_limit("1")
    task.set_memory_request(memory_request).set_memory_limit(memory_limit)
    kubernetes.mount_pvc(task, pvc_name="cv-data", mount_path="/data")
    return task


@dsl.pipeline(name=PIPELINE_NAME, description=DESCRIPTION)
def cv_shard_pipeline():
    split = attach(split_dataset(), "500m", "512Mi", "2Gi")
    packed = attach(
        pack_shards(split_file=split.outputs["split_file"]),
        "500m",
        "1Gi",
        "2Gi",
    )
    trained = attach(train(manifest=packed.outputs["manifest"]), "500m", "512Mi", "1Gi")
    scored = attach(evaluate(manifest=packed.outputs["manifest"]), "500m", "512Mi", "1Gi")
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


def submit(path=PACKAGE):
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
    run_body = kfp_server_api.V2beta1Run(
        display_name=run_name,
        description=DESCRIPTION,
        experiment_id=experiment.experiment_id,
        pipeline_version_reference=reference,
        plugins_input={"mlflow": {"experiment_name": MLFLOW_EXPERIMENT}},
    )
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


if __name__ == "__main__":
    submit()
