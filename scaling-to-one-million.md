# From 20,000 images to one million

The 20,000-image pipeline is the same path a million-image run uses. Split, pack, TrainJob, and evaluate stay as they are. Image count, epoch count, and the wait limit are run settings. Copy the scripts to `/data/code` before starting the pipeline. The pods do not read the git checkout.

The million-image corpus is already on the shared volume. This file is the handoff for the next change. It is not a request to start another run.

## What one epoch means

An epoch is one full pass over the training shards. ResNet18 is created once, from random weights, and the optimizer keeps those weights. The next epoch reads the shards again and keeps updating the same model. One epoch is the demo default (`EPOCHS = 1` in `train.py`) so the cluster test finishes in minutes. It is not a property of image classification or of this network.

The synthetic shapes are easy. The 20,000-image run reached about 0.77 train accuracy and 0.83 test accuracy in one epoch. The million-image run, ten epochs, reached about 0.99 in epoch 1 and 1.0 from epoch 2 on. The held-out test set also scored 1.0. Train and test paths did not overlap (800,000 and 200,000 unique files, 200,000 / 50,000 per class). The score is the drawings, not a leaked split. More epochs on the same generator will stay at 1.0.

The pipeline `epochs` parameter is passed into the TrainJob as `EPOCHS`. `torchrun.sh` adds `--epochs` only when that variable is set. Otherwise `train.py` uses `EPOCHS = 1`.

The reported `loss` and `train_accuracy` are the last epoch, combined across both GPUs. `images_per_sec` is every image processed during the fit, divided by the wall time. Rank 0 also writes `history`: loss, accuracy, macro precision, recall, and F1, and the per-class scores for that epoch. The train pipeline step logs each point with `mlflow.log_metric(..., step=epoch)` when `MLFLOW_RUN_ID` is present. It does not call `mlflow.start_run()`. The evaluate step logs the test confusion matrix and the per-class test scores. Training pods still do not call MLflow. Each pod runs `gpu_sample.py`, and `submit_trainjob.py` logs that series as `system/node_<n>/gpu_<i>_utilization_percentage` and `system/node_<n>/gpu_<i>_memory_usage_megabytes`. Turn on **Allow pipeline tasks to log directly to MLflow** or the epoch curve, the matrix image, and the GPU series are not written.

## Knobs

| Change | Where | Million-image value |
| --- | --- | --- |
| Image count | `generate_images.py --count` | Already 1,000,000 under `/data/raw`. The notebook asks for 20,000 and stops if that would delete a larger corpus. A matching stamp fills gaps. `--replace` is required to shrink. |
| Epochs | pipeline parameter `epochs` | Default 1. The finished million run used 10. Copy `train.py` and `torchrun.sh` to `/data/code`. |
| Wait limit | `TIMEOUT_SEC` in `submit_trainjob.py` | Default 3600 seconds. The 10-epoch million fit took about 75 minutes, so raise this before another full fit. Copy this file to `/data/code` too. |
| Trial size and outputs | pipeline `max_per_class`, `shard_dir`, `metrics_dir`, `checkpoint` | `max_per_class` 0 uses every image. A trial must write shards, metrics, the split, and the checkpoint under a different directory, such as `/data/subset`. Pack deletes `shard-*.tar` inside `<shard_dir>/train` and `<shard_dir>/test`. |

Leave these alone:

- Images per shard stay at 2,000. One million images become 400 train shards and 100 test shards, which divide evenly across the two GPUs.
- `numNodes` stays 2. Shard shuffle stays off. Each rank must see the same shard list, and `split_by_node` does the split.
- The pipeline name stays `overhead-shard-training`. The parameterized graph is version `v2`. Version `v1` is the previous graph and cannot be edited in place.
- The shared volume is 100Gi. The current 20,000 images use about 205MB of JPEGs and 196MB of shards, so one million is about 20GB including shards.

## Time

The million-image fit processed about 1,800 images per second, same as the 20,000-image fit. Ten epochs over 800,000 training images took about 75 minutes. Scoring 200,000 test images took about 96 seconds. A one-hour wait is too short for that fit. Generation is a workbench job, not a pipeline step, and the million JPEGs are already written.

## A 100,000-image trial

The million-image corpus stays on `/data/raw`. `max_per_class 25000` keeps the first 25,000 JPEGs of each class, 100,000 images in total. Pack deletes `shard-*.tar` inside the train and test folders of the shard directory it is given, so the trial uses its own directories:

```bash
python pipeline.py \
  --max-per-class 25000 \
  --epochs 3 \
  --shard-dir /data/subset-100k/shards \
  --metrics-dir /data/subset-100k/metrics \
  --checkpoint /data/subset-100k/models/resnet18.pt
```

Copy `train.py` to `/data/code` before starting. The pods run that copy.

## Later improvements

Do these in a later iteration, not as part of turning the count up:

- Register the checkpoint only after a threshold on `test_accuracy`.
- Put the project under Kueue if the GPUs are shared.
- Mixed precision, if a profile says the fit is compute bound.
