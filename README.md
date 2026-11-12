# CV shard training

OpenShift AI 3.5 demo of an overhead image classifier. A notebook generates synthetic 256×256 JPEGs of trucks, boats, planes, and buildings. A Kubeflow pipeline then splits that corpus, packs WebDataset shards, submits distributed training, and evaluates the held-out shards.

The demo defaults are 20,000 images and one training epoch. That is a size and time choice, not a limit of ResNet18. A million-image run already finished on the cluster. [scaling-to-one-million.md](scaling-to-one-million.md) records that result and the knobs for another one.

## How the pieces fit

The pipeline orchestrates. It does not host the GPUs.

1. `split-dataset` writes a stratified 80/20 split.
2. `pack-shards` writes WebDataset tar shards onto the shared volume. Each sample is `000123.jpg` plus `000123.cls`.
3. `train` submits a Kubeflow Trainer v2 TrainJob and waits. Two nodes, one GPU each, run `train.py` with torchrun. Rank 0 writes `/data/models/resnet18.pt` and `/data/metrics/train-metrics.json`.
4. `evaluate` submits a second TrainJob, loads that checkpoint, and writes `/data/metrics/test-metrics.json`.

Each pipeline step reads the JSON it needs and publishes scalars with `metrics.log_metric`. The platform copies those scalars into a nested MLflow run, one run per step, under the experiment `overhead-shard-training`. The parent MLflow run stays empty. Training pods do not call MLflow.

The TrainJob uses runtime `torch-distributed-cuda130-torch211-py312`. WebDataset is imported from `/data/pydeps`.

## What a successful run records

Pipeline `overhead-shard-training`. The pipeline experiment and the MLflow experiment share that name. The numbers below are the 20,000-image run, which used version `v1`. The current graph is version `v2`, because a pipeline version cannot be edited in place. On that run the nested runs held:

| Step | Metrics |
| --- | --- |
| `split-dataset` | `train_count`, `test_count` |
| `pack-shards` | `image_count`, `train_shards`, `test_shards` |
| `train` | `loss`, `train_accuracy`, `images_per_sec` |
| `evaluate` | `test_loss`, `test_accuracy`, `test_images` |

`loss` and `train_accuracy` are the last epoch. With the default of one epoch, that is the whole fit. `epoch_seconds` in the JSON is the wall time of the fit. The train JSON also keeps a `history` list, one loss and accuracy per epoch. The train step logs those points onto the nested MLflow run when the platform has injected `MLFLOW_RUN_ID`. Until `injectUserEnvVars` is turned on, MLflow still shows the single last-epoch scalars.

## Assumptions

The cluster already has OpenShift AI 3.5, including MLflow, an RWX storage class, GPU nodes, and the Trainer runtime named above. This demo does not call MaaS, and it does not put the project under Kueue. The checkpoint stays on the shared volume. Model registry promotion is later work.

The manifests use storage classes `efs-csi` (shared data) and `gp3-csi` (workbench and pipeline database). They also name an S3 bucket, an MLflow URL, and the dashboard host for one sandbox. Change those before applying them on another cluster.

## What is included

| Path | Purpose |
| --- | --- |
| `demo.ipynb` | Generate the images, show a 4×4 preview, launch the pipeline |
| `generate_images.py` | Labeled 256×256 JPEGs, four overhead classes. Default count is 20,000 |
| `split_dataset.py` | Stratified 80/20 split, written as a JSON list the pack step consumes |
| `pack_shards.py` | Pack that split into train and test shards and write the manifest |
| `train.py` | ResNet18 from scratch. `--phase train` fits `EPOCHS` and saves a checkpoint. `--phase eval` scores the test shards |
| `submit_trainjob.py` | Create one TrainJob for the train phase or the eval phase and wait for it |
| `torchrun.sh` | Launch `train.py` with the phase and the `PET_*` values injected by the Trainer runtime |
| `pipeline.py` | Registers pipeline `overhead-shard-training` version `v2`, then starts a run |
| `scaling-to-one-million.md` | What to change for a million images and more epochs |
| `manifests/` | Project, volumes, workbench, pipeline server, workflow trust config, and TrainJob RBAC |

## Setup

1. Create secret `aws-s3-pipelines` in the project with keys `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`. Set the bucket name in `manifests/dspa.yaml` and `manifests/workflow-controller-config.yaml`. Do not commit the secret.
2. Apply `manifests/namespace.yaml`, `pvc.yaml`, `workbench.yaml`, `workflow-controller-config.yaml`, `dspa.yaml`, and `pipeline-rbac.yaml`. Leave MLflow on the platform defaults (`integrationMode` stays unset, so it autodetects, and `injectUserEnvVars` stays false). The workflow config keeps the S3 artifact store and sets `SSL_CERT_FILE=/kfp/certs/ca.crt` so the pipeline driver trusts the cluster service CA when it opens each nested MLflow run. Apply that config before the pipeline server. Do not edit the generated `ds-pipeline-workflow-controller-dspa` config. The operator replaces it.
3. Put this repo in the workbench home directory. The shared volume `cv-data` is mounted at `/opt/app-root/src/data` in the workbench and at `/data` on the pipeline and training pods.
4. Copy `split_dataset.py`, `pack_shards.py`, `train.py`, `submit_trainjob.py`, and `torchrun.sh` to `/opt/app-root/src/data/code`. The pods run those copies.
5. Install WebDataset onto the shared volume. The workbench pip index does not carry it:

   ```bash
   pip install --index-url https://pypi.org/simple --target /opt/app-root/src/data/pydeps --no-deps webdataset==0.2.100 braceexpand
   ```

6. Run `demo.ipynb` from the top. The generator stamp is `overhead-silhouettes-v1`. A later run with the same stamp, class list, and count skips JPEGs that are already there.
7. Open **Develop and train → Pipelines**. The next run is listed under pipeline `overhead-shard-training`, version `v2`, with steps `split-dataset`, `pack-shards`, `train`, and `evaluate`. Open the nested MLflow runs to see the metrics. The parent run has none. Version `v1` remains the earlier graph.
