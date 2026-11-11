# CV shard training

OpenShift AI 3.5 demo. A notebook generates synthetic images, one pipeline packs them into WebDataset tar shards, and that pipeline starts a two-node TrainJob.

WebDataset shards are tar files. Each sample is `000123.jpg` plus `000123.cls`. The image bytes stay on the shared volume. The pipeline stores only the small manifest.

## Assumptions

The cluster already has OpenShift AI 3.5, including:

- MaaS
- MLflow
- Kueue
- an RWX storage class
- GPU nodes and the Trainer runtime `torch-distributed-cuda130-torch211-py312`

This demo does not call MaaS, and it does not put the project under Kueue. Training metrics are the pipeline artifact written by the TrainJob.

The manifests use storage classes `efs-csi` (shared data) and `gp3-csi` (workbench and pipeline database). They also name an S3 bucket, an MLflow URL, and the dashboard host for one sandbox. Change those before applying them on another cluster.

## What is included

| Path | Purpose |
| --- | --- |
| `demo.ipynb` | Generate the images, show one sample per class, launch the pipeline |
| `generate_images.py` | 20,000 labeled 256×256 JPEGs, ten shape classes |
| `pack_shards.py` | Pack those JPEGs into WebDataset `.tar` shards and write the manifest |
| `train.py` | ResNet18 from scratch, DDP, one epoch, WebDataset `split_by_node` / `split_by_worker` |
| `submit_trainjob.py` | Create one TrainJob and wait for it |
| `torchrun.sh` | Launch `train.py` with the `PET_*` values injected by the Trainer runtime |
| `pipeline.py` | Kubeflow Pipelines graph: `pack_shards`, then `train` |
| `manifests/` | Project, volumes, workbench, pipeline server, and TrainJob RBAC |

## Setup

1. Create secret `aws-s3-pipelines` in the project with keys `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`. Set the bucket name in `manifests/dspa.yaml`. Do not commit the secret.
2. Apply `manifests/namespace.yaml`, `pvc.yaml`, `workbench.yaml`, `dspa.yaml`, and `pipeline-rbac.yaml`.
3. Put this repo in the workbench home directory. The shared volume `cv-data` is mounted at `/opt/app-root/src/data` in the workbench and at `/data` on the pipeline and training pods.
4. Copy `pack_shards.py`, `train.py`, `submit_trainjob.py`, and `torchrun.sh` to `/opt/app-root/src/data/code`.
5. Install WebDataset onto the shared volume. The workbench pip index does not carry it:

   ```bash
   pip install --index-url https://pypi.org/simple --target /opt/app-root/src/data/pydeps --no-deps webdataset==0.2.100 braceexpand
   ```

6. Run `demo.ipynb` from the top.
7. Open **Develop and train → Pipelines → Runs**.

The TrainJob uses runtime `torch-distributed-cuda130-torch211-py312`, two nodes, one GPU each. `webdataset` is read from `/data/pydeps` through `PYTHONPATH`.
