# From 20,000 images to one million

The 20,000-image pipeline is the same path a million-image run uses. Split, pack, TrainJob, and evaluate stay as they are. Change the corpus size, the epoch count, and the wait limit. Then copy the changed scripts to `/data/code` before starting the pipeline. The pods do not read the git checkout.

This file is the handoff for that next iteration. It is not a request to start the run.

## What one epoch means

An epoch is one full pass over the training shards. ResNet18 is created once, from random weights, and the optimizer keeps those weights. The next epoch reads the shards again and keeps updating the same model. One epoch is the demo default (`EPOCHS = 1` in `train.py`) so the cluster test finishes in minutes. It is not a property of image classification or of this network.

The synthetic shapes are easy, which is why one pass already reaches roughly 0.77 train accuracy and 0.83 test accuracy. A longer fit is still more passes over the same data. `torchrun.sh` does not pass `--epochs`. The value that runs is the default in `train.py` on the shared volume.

The reported `loss` and `train_accuracy` are the last epoch, combined across both GPUs. `images_per_sec` is every image processed during the fit, divided by the wall time. MLflow still receives one number for each of those names. A curve of loss by epoch is a later change: write the history into the metrics JSON, then log each point from the pipeline step. Do not point the TrainJob pods at MLflow.

## Knobs

| Change | Where | Million-image value |
| --- | --- | --- |
| Image count | `generate(1000000, RAW, workers=2)` in `demo.ipynb` | 1,000,000 JPEGs. The stamp matches the images already on disk, so the existing 20,000 stay and the rest are written. The workbench has 2 CPUs, so `workers=2` is the useful setting. |
| Epochs | `EPOCHS` in `train.py` | More than 1. Copy `train.py` to `/data/code`. |
| Wait limit | `TIMEOUT_SEC` in `submit_trainjob.py` | Above the fit time. The default is 3600 seconds. Copy this file to `/data/code` too. |

Leave these alone:

- Images per shard stay at 2,000. One million images become 400 train shards and 100 test shards, which divide evenly across the two GPUs.
- `numNodes` stays 2. Shard shuffle stays off. Each rank must see the same shard list, and `split_by_node` does the split.
- The pipeline graph stays `overhead-shard-training` version `v1`. Count and epochs are not pipeline parameters.
- The shared volume is 100Gi. The current 20,000 images use about 205MB of JPEGs and 196MB of shards, so one million is about 20GB including shards.

## Time

The last 20,000-image fit processed about 1,800 images per second. The training split is 80 percent, so one epoch of a million images is on the order of 7 minutes of GPU time before EFS slowdown. Five epochs fit inside the current one-hour wait. Ten epochs run past it, and a cold read of 400 shards can be slower than the small run. Raise `TIMEOUT_SEC` before that run. Generation of the extra images is a workbench job, not a pipeline step, and it is the long CPU task.

## Later improvements

Do these in a later iteration, not as part of turning the count up:

- Log each epoch to MLflow from the train pipeline step, using the nested run the platform already creates. That needs `injectUserEnvVars` on the pipeline server. The worker pods still do not get the tracking URI or the service CA.
- Register the checkpoint only after a threshold on `test_accuracy`.
- Put the project under Kueue if the GPUs are shared.
- Mixed precision, if a profile says the fit is compute bound.
