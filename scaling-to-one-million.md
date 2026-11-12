# From 20,000 images to one million

The 20,000-image pipeline is the same path a million-image run uses. Split, pack, TrainJob, and evaluate stay as they are. Image count, epoch count, and the wait limit are run settings. Copy the scripts to `/data/code` before starting the pipeline. The pods do not read the git checkout.

The million-image corpus is already on the shared volume. This file is the handoff for the next change. It is not a request to start another run.

## What one epoch means

An epoch is one full pass over the training shards. ResNet18 is created once, from random weights, and the optimizer keeps those weights. The next epoch reads the shards again and keeps updating the same model. One epoch is the demo default (`EPOCHS = 1` in `train.py`) so the cluster test finishes in minutes. It is not a property of image classification or of this network.

The synthetic shapes are easy. The 20,000-image run reached about 0.77 train accuracy and 0.83 test accuracy in one epoch. The million-image run, ten epochs, reached about 0.99 in epoch 1 and 1.0 from epoch 2 on. The held-out test set also scored 1.0. Train and test paths did not overlap (800,000 and 200,000 unique files, 200,000 / 50,000 per class). The score is the drawings, not a leaked split. More epochs on the same generator will stay at 1.0.

The pipeline `epochs` parameter is passed into the TrainJob as `EPOCHS`. `torchrun.sh` adds `--epochs` only when that variable is set. Otherwise `train.py` uses `EPOCHS = 1`.

The reported `loss` and `train_accuracy` are the last epoch, combined across both GPUs. `images_per_sec` is every image processed during the fit, divided by the wall time. Rank 0 also writes `history`, one loss and accuracy per epoch. The train pipeline step logs each point with `mlflow.log_metric(..., step=epoch)` when `MLFLOW_RUN_ID` is present. It does not call `mlflow.start_run()`. Training pods still do not call MLflow.

## Knobs

| Change | Where | Million-image value |
| --- | --- | --- |
| Image count | `generate_images.py --count` | Already 1,000,000 under `/data/raw`. The notebook still generates 20,000 so a click does not write another million. The stamp matches, so a rerun fills gaps instead of rewriting. |
| Epochs | pipeline parameter `epochs` | Default 1. The finished million run used 10. Copy `train.py` and `torchrun.sh` to `/data/code`. |
| Wait limit | `TIMEOUT_SEC` in `submit_trainjob.py` | Default 3600 seconds. The 10-epoch million fit took about 75 minutes, so raise this before another full fit. Copy this file to `/data/code` too. |
| Trial size and outputs | pipeline `max_per_class`, `shard_dir`, `metrics_dir`, `checkpoint` | `max_per_class` 0 uses every image. A trial must write shards, metrics, the split, and the checkpoint under a different directory, such as `/data/subset`. Pack deletes `shard-*.tar` in the shard directory it is given. |

Leave these alone:

- Images per shard stay at 2,000. One million images become 400 train shards and 100 test shards, which divide evenly across the two GPUs.
- `numNodes` stays 2. Shard shuffle stays off. Each rank must see the same shard list, and `split_by_node` does the split.
- The pipeline name stays `overhead-shard-training`. The parameterized graph is version `v2`. Version `v1` is the previous graph and cannot be edited in place.
- The shared volume is 100Gi. The current 20,000 images use about 205MB of JPEGs and 196MB of shards, so one million is about 20GB including shards.

## Time

The million-image fit processed about 1,800 images per second, same as the 20,000-image fit. Ten epochs over 800,000 training images took about 75 minutes. Scoring 200,000 test images took about 96 seconds. A one-hour wait is too short for that fit. Generation is a workbench job, not a pipeline step, and the million JPEGs are already written.

## Later improvements

Do these in a later iteration, not as part of turning the count up:

- Turn on `injectUserEnvVars` for the pipeline server so the train step receives `MLFLOW_RUN_ID` and the per-epoch history becomes a curve. The logging code is already in the train step. The worker pods still do not get the tracking URI or the service CA. The flag is still off on the cluster.
- Register the checkpoint only after a threshold on `test_accuracy`.
- Put the project under Kueue if the GPUs are shared.
- Mixed precision, if a profile says the fit is compute bound.
