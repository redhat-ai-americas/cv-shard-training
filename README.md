# CV shard training

OpenShift AI 3.5 demo of an overhead image classifier. A workbench draws synthetic 256×256 JPEGs of trucks, boats, planes, and buildings. A pipeline then splits that corpus, packs it into WebDataset shards, trains ResNet18 on two GPUs, and scores the held-out shards. Metrics, a confusion matrix, and GPU utilization land in MLflow.

The defaults are 20,000 images and one epoch. That is a time choice, not a limit of the model. A million-image, ten-epoch run on this pattern trained at about the same 1,800 images per second. Reading the million individual files into shards took about 2 hours 15 minutes and used no GPU. The fit itself took about 75 minutes, with both GPUs mostly busy. [scaling-to-one-million.md](scaling-to-one-million.md) records that run.

This is the path a customer follows, from an empty project to a scored model. The manifests match one sandbox. Change the settings in the next section before applying them anywhere else.

## What the cluster must already have

OpenShift AI 3.5, with these platform pieces turned on:

- The MLflow Operator component, and an `mlflow` custom resource. A small install uses SQLite and a volume for artifacts. See [Install and configure MLflow](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.5/html/working_with_mlflow/installing-mlflow_mlflow).
- Data science pipelines, so a project can create a pipeline server.
- A Kubeflow Trainer v2 `ClusterTrainingRuntime` named `torch-distributed-cuda130-torch211-py312`. TrainJobs use that runtime. See [Run Kubeflow Trainer v2 workloads](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.5/html/working_with_distributed_workloads/running-kubeflow-trainerv2_distributed-workloads).
- At least two GPU nodes, and an RWX storage class for the shared data volume.
- A hardware profile the workbench can use. This sandbox uses `large-cpu-profile` in `redhat-ods-applications`.

The demo does not call MaaS, and it does not put the project under Kueue. The checkpoint stays on the shared volume. Registering it in the model registry is later work.

## Settings to change on another cluster

| Setting | Where | This sandbox |
| --- | --- | --- |
| Project | `manifests/namespace.yaml` | `cv-shard-training` |
| Shared volume class and size | `manifests/pvc.yaml` `cv-data` | `efs-csi`, 100Gi, ReadWriteMany |
| Workbench volume class and size | `manifests/pvc.yaml` `cv-workbench` | `gp3-csi`, 20Gi, ReadWriteOnce |
| Pipeline database class | `manifests/dspa.yaml` | `gp3-csi`, 10Gi MariaDB |
| S3 bucket, region, host | `manifests/dspa.yaml` and `manifests/workflow-controller-config.yaml` | `cv-shard-pipelines-9j696`, `us-east-2` |
| S3 credentials | secret `aws-s3-pipelines`, keys `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` | Not in git |
| Dashboard host | `pipeline.py` `DASHBOARD`, and `manifests/workbench.yaml` `MLFLOW_TRACKING_URI` | `rh-ai.apps.ocp.9j696.sandbox5421.opentlc.com` |
| Pipeline API | `pipeline.py` `PIPELINE_HOST` | `https://ds-pipeline-dspa.cv-shard-training.svc.cluster.local:8443` |
| Workbench image | `manifests/workbench.yaml` | `pytorch:3.5` from the OpenShift AI image registry |
| Workbench size | `manifests/workbench.yaml` | 2 CPU, 8Gi. The notebook URL path includes the project and workbench name |
| Trainer runtime | `submit_trainjob.py` `RUNTIME` | `torch-distributed-cuda130-torch211-py312` |
| TrainJob shape | `submit_trainjob.py` `build_train_job` | 2 nodes, 1 GPU, 4 CPU, and 24Gi each |
| Pipeline task image | `pipeline.py` `BASE_IMAGE` | `s2i-generic-data-science-notebook:3.5` |
| Pipeline version | `pipeline.py` `VERSION_NAME` | `v2`. A version cannot be edited. The next graph change needs `v3` |

`pipeline.py` sets `verify_ssl=False` for the in-cluster pipeline API. The workflow controller still needs the cluster CA. That is the `SSL_CERT_FILE=/kfp/certs/ca.crt` entry in `manifests/workflow-controller-config.yaml`. Apply that ConfigMap before the pipeline server. Do not edit the generated ConfigMap `ds-pipeline-workflow-controller-dspa`. The operator replaces it.

## Dependencies

The workbench image already includes Python 3.12, PyTorch, Pillow, NumPy, matplotlib, and MLflow. Starting with OpenShift AI 3.4, the standard workbench images include the MLflow SDK. The pipeline task image includes MLflow and matplotlib as well. This repo adds only what those images do not carry.

| Package | Where it is installed | Pin |
| --- | --- | --- |
| `kfp` | Workbench, by the last notebook cell, from `https://pypi.org/simple` | `2.17.0` |
| `webdataset` and `braceexpand` | Shared volume `/data/pydeps`, imported by the TrainJob through `PYTHONPATH` | `webdataset==0.2.100` |
| CUDA, PyTorch, `nvidia-smi` | The Trainer runtime image | The runtime named above |

No other pip installs are required. Do not commit a virtualenv or the compiled pipeline YAML. `.gitignore` already excludes `cv_shard_pipeline.yaml`.

## From an empty project to a scored model

1. Create secret `aws-s3-pipelines` in the project. Do not commit it.
2. Apply the manifests in this order: `namespace.yaml`, `pvc.yaml`, `workbench.yaml`, `workflow-controller-config.yaml`, `dspa.yaml`, `pipeline-rbac.yaml`.

   ```bash
   oc apply -f manifests/namespace.yaml
   oc apply -f manifests/pvc.yaml
   oc apply -f manifests/workbench.yaml
   oc apply -f manifests/workflow-controller-config.yaml
   oc apply -f manifests/dspa.yaml
   oc apply -f manifests/pipeline-rbac.yaml
   ```

   Leave `integrationMode` unset so MLflow autodetects. Do not full-apply `dspa.yaml` over a pipeline server that is already running. The file does not contain every field the live server may have grown.
3. In the dashboard, open the project, then **Manage pipeline server → Advanced settings**, and turn on **Allow pipeline tasks to log directly to MLflow**. That is `injectUserEnvVars`. The pipeline driver creates one nested MLflow run per step and injects `MLFLOW_TRACKING_URI`, `MLFLOW_RUN_ID`, and the workspace into that step. The task code logs into that run. It does not call `mlflow.start_run()`.
4. Open the workbench **CV shard training**. The `opendatahub.io/mlflow-instance: mlflow` annotation is what OpenShift AI uses to inject MLflow into a workbench. See [Track experiments with MLflow in workbenches](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.5/html/working_with_mlflow/tracking-experiments-with-mlflow-in-workbenches). Dashboard login sits in front of the notebook. The empty notebook token in the manifest is the usual workbench setting, not an extra password.
5. Put this repo in the workbench home, `/opt/app-root/src`. The shared volume `cv-data` is `/opt/app-root/src/data` in the workbench and `/data` on the pipeline and training pods.
6. Copy the scripts the pods actually run. Editing the git checkout is not enough.

   ```bash
   mkdir -p /opt/app-root/src/data/code
   cp split_dataset.py pack_shards.py train.py submit_trainjob.py gpu_sample.py torchrun.sh \
     /opt/app-root/src/data/code/
   ```

7. Install WebDataset onto the shared volume. The workbench pip index does not carry it.

   ```bash
   pip install --index-url https://pypi.org/simple \
     --target /opt/app-root/src/data/pydeps --no-deps \
     webdataset==0.2.100 braceexpand
   ```

8. Run `demo.ipynb` from the top.
   - The first code cell writes 20,000 JPEGs under `/data/raw`, 5,000 of each class. The generator stamp is `overhead-silhouettes-v1`. A later run with the same stamp fills gaps. If the folder already has more than 20,000 images, the cell stops. Shrinking it requires `generate(..., replace=True)` or `python generate_images.py --count 20000 --replace`.
   - The next cell shows a 4×4 preview.
   - The last cell installs `kfp` and calls `submit()`. That registers pipeline `overhead-shard-training` version `v2` if needed, then starts a run with the defaults: every image in `/data/raw`, one epoch, shards under `/data/shards`, metrics under `/data/metrics`, checkpoint `/data/models/resnet18.pt`.
9. Open **Develop and train → Pipelines**. The run has four steps: `split-dataset`, `pack-shards`, `train`, `evaluate`. Pack deletes `shard-*.tar` inside `<shard_dir>/train` and `<shard_dir>/test` before it writes the new shards. A subset run must pass a different shard directory, metrics directory, and checkpoint. `pipeline.py` refuses a subset that targets the default paths.

   ```bash
   python pipeline.py \
     --max-per-class 5000 \
     --epochs 3 \
     --shard-dir /data/subset-20k/shards \
     --metrics-dir /data/subset-20k/metrics \
     --checkpoint /data/subset-20k/models/resnet18.pt
   ```

   `--max-per-class 5000` keeps the first 5,000 JPEGs of each class, 20,000 images in total. `0` means every image.
10. Open **Develop and train → Experiments (MLflow)**, experiment `overhead-shard-training`. The parent run is empty. Open the nested run for `train` and the nested run for `evaluate`.

`run-million.sh` is the unattended form of the full corpus: it generates 1,000,000 images and then runs `pipeline.py` with the defaults, including one epoch and a rewrite of `/data/shards`. It is not the ten-epoch fit. Use the scaling note for that.

## What a successful run records

The train curve is the training set, one point per epoch. The confusion matrix and the `test_*` scores are the held-out 20 percent. Rows of the matrix are the true class, columns are the predicted class, in the order `truck`, `boat`, `plane`, `building`.

| Step | What shows up |
| --- | --- |
| `split-dataset` | `train_count`, `test_count` |
| `pack-shards` | `image_count`, `train_shards`, `test_shards` |
| `train` | One point per epoch for `loss`, `train_accuracy`, macro precision, recall, and F1, and the same three scores per class. `images_per_sec` is the whole fit. System metrics hold `system/node_<n>/gpu_<i>_utilization_percentage` and `system/node_<n>/gpu_<i>_memory_usage_megabytes`. |
| `evaluate` | `test_loss`, `test_accuracy`, `test_images`, macro and weighted precision, recall, and F1, per-class precision, recall, F1, and support, and one `test_cm_<true>_<predicted>` count per cell. Artifacts `confusion_matrix.png`, `confusion_matrix.json`, and `confusion_matrix.txt` are on the nested run when direct MLflow logging is on. The eval TrainJob records the same GPU series. Eval is often shorter than the 15 second sample interval, so that series can be a single startup point. |

Scalar `test_*` metrics also go through `metrics.log_metric`, which the platform copies even when the checkbox is off. The epoch curve, the matrix image, and the GPU series need the checkbox. A missing `nvidia-smi`, or a failed MLflow write of the GPU series or the matrix image, is printed and does not fail the step.

GPU samples are taken on the training pods. Those pods do not call MLflow. The Trainer service account has no MLflow role, and turning on MLflow's own system sampler inside the pipeline task would record the idle orchestrator, which has no GPU. `gpu_sample.py` runs `nvidia-smi` every 15 seconds. After the TrainJob finishes, `submit_trainjob.py` logs the samples on the nested run with the `system/` names MLflow groups as System metrics.

On the 20,000-image check, one GPU reached 99 percent and the other 82 percent, both holding about 7,043 MB. On the million-image fit, the minute-by-minute medians were about 91 percent and 86 percent, memory stayed near 7,042 MB, and throughput stayed at about 1,793 images per second. The dips in that chart are one-minute samples of a small 256×256 batch, shard reads, and the two GPUs waiting on each other. They are not the flat idle line of a loader that opens a million objects during training.

## Training defaults

| Knob | Value | Where |
| --- | --- | --- |
| Classes | truck, boat, plane, building | `generate_images.py` `CLASSES` |
| Image size | 256×256 JPEG, quality 85 | `generate_images.py` |
| Split | 80/20, seed 17 | `split_dataset.py` |
| Images per shard | 2,000 | `pack_shards.py` |
| Epochs | 1, unless the run passes `--epochs` | `train.py`, `pipeline.py` |
| Batch size | 128 | `train.py` |
| Loader workers | 2 per GPU | `train.py` |
| Learning rate | 0.05 | `train.py` |
| Model | ResNet18, trained from scratch | `train.py` |
| Checkpoint | `/data/models/resnet18.pt` | Rank 0 |
| Train wait | 3,600 seconds | `submit_trainjob.py` `TIMEOUT_SEC` |

Shard shuffle is off. Each rank must see the same shard list. `split_by_node` divides it across the two GPUs.

## What is included

| Path | Purpose |
| --- | --- |
| `demo.ipynb` | Generate the images, show a preview, launch the default pipeline run |
| `generate_images.py` | Draw the labeled JPEGs. Default count is 20,000 |
| `split_dataset.py` | Stratified split, as a JSON list the pack step consumes |
| `pack_shards.py` | Pack that split into train and test shards and write the manifest |
| `train.py` | ResNet18. `--phase train` fits and saves the checkpoint. `--phase eval` scores the test shards |
| `submit_trainjob.py` | Create one TrainJob, wait for it, and log its GPU samples |
| `gpu_sample.py` | Sample `nvidia-smi` on each training pod until that pod exits |
| `torchrun.sh` | Launch `train.py` with the `PET_*` values the Trainer runtime injects |
| `pipeline.py` | Register the pipeline version, if needed, and start a run |
| `run-million.sh` | Generate 1,000,000 images and start the default one-epoch pipeline |
| `tests/test_handoff.py` | Guard and metric-name checks. No GPU and no cluster |
| `scaling-to-one-million.md` | The million-image result and the knobs for another one |
| `manifests/` | Project, volumes, workbench, pipeline server, workflow CA, and TrainJob RBAC |

Run the local checks with `python -m unittest tests.test_handoff`.

## After the first run

- Compare runs under **Develop and train → Experiments (MLflow)**. The dashboard view is read-only. See [Track and compare MLflow experiments](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.5/html/working_with_mlflow/track-and-compare-mlflow-experiments_mlflow).
- The checkpoint to promote later is the path passed as `checkpoint`. The default run writes `/data/models/resnet18.pt`.
- Copy any changed script under `/data/code` before the next run. Pipeline version `v2` is already uploaded. A change to the graph in `pipeline.py` needs a new `VERSION_NAME`. A change only to the scripts on the volume does not.
