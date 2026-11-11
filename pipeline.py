"""Two-step pipeline: pack WebDataset shards, then submit the TrainJob."""

import os
import time

from kfp import compiler, dsl, kubernetes
from kfp.client import Client
from kfp.dsl import Artifact, Input, Output

BASE_IMAGE = (
    "image-registry.openshift-image-registry.svc:5000/"
    "redhat-ods-applications/s2i-generic-data-science-notebook:3.5"
)
PYPI = ["https://pypi.org/simple"]
PIPELINE_HOST = "https://ds-pipeline-dspa.cv-shard-training.svc.cluster.local:8443"
DASHBOARD = "https://rh-ai.apps.ocp.9j696.sandbox5421.opentlc.com"
PACKAGE = "/opt/app-root/src/cv_shard_pipeline.yaml"


@dsl.component(base_image=BASE_IMAGE, pip_index_urls=PYPI)
def pack_shards(manifest: Output[Artifact]):
    import subprocess
    import sys

    subprocess.check_call(
        [
            sys.executable,
            "/data/code/pack_shards.py",
            "--raw",
            "/data/raw",
            "--shards",
            "/data/shards",
            "--manifest",
            manifest.path,
        ]
    )


@dsl.component(base_image=BASE_IMAGE, pip_index_urls=PYPI)
def train(manifest: Input[Artifact], metrics: Output[Artifact]):
    import subprocess
    import sys

    subprocess.check_call(
        [
            sys.executable,
            "/data/code/submit_trainjob.py",
            "--manifest",
            manifest.path,
            "--metrics",
            metrics.path,
        ]
    )


@dsl.pipeline(name="cv-shard-training", description="Pack images into shards and train ResNet18")
def cv_shard_pipeline():
    packed = pack_shards()
    packed.set_caching_options(False)
    packed.set_cpu_request("500m").set_cpu_limit("1")
    packed.set_memory_request("1Gi").set_memory_limit("2Gi")
    kubernetes.mount_pvc(packed, pvc_name="cv-data", mount_path="/data")

    trained = train(manifest=packed.outputs["manifest"])
    trained.set_caching_options(False)
    trained.set_cpu_request("500m").set_cpu_limit("1")
    trained.set_memory_request("512Mi").set_memory_limit("1Gi")
    kubernetes.mount_pvc(trained, pvc_name="cv-data", mount_path="/data")


def compile_pipeline(path=PACKAGE):
    compiler.Compiler().compile(cv_shard_pipeline, path)
    return path


def submit(path=PACKAGE):
    compile_pipeline(path)
    token = open("/var/run/secrets/kubernetes.io/serviceaccount/token", encoding="utf-8").read()
    client = Client(
        host=PIPELINE_HOST,
        existing_token=token,
        namespace="cv-shard-training",
        verify_ssl=False,
    )
    run_name = "cv-shard-" + time.strftime("%H%M%S")
    run = client.create_run_from_pipeline_package(
        path,
        arguments={},
        run_name=run_name,
        experiment_name="cv-shard-training",
    )
    print(f"run_name={run_name}")
    print(f"run_id={run.run_id}")
    print(f"Open {DASHBOARD} and go to CV shard training, Develop and train, Pipelines, Runs.")
    return run


if __name__ == "__main__":
    submit()
