# Run one million images

Same pipeline as the 20,000-image demo. You draw the corpus first. A fresh checkout does not contain the JPEGs, and a new cluster does not have them on the volume.

Do this after the workbench, the shared volume, WebDataset, and `/data/code` are in place. Those steps are in the [README](README.md). Copy any script you change into `/data/code` before you start. The pods read that copy, not the git checkout.

## Draw the images

Generation runs on the workbench. It is Pillow drawing, so a GPU does not help. On the demo workbench, 2 CPUs and `--workers 2`, a timed sample wrote about 70 images a second to the shared volume. At that rate:

| Count | Drawing time | Disk |
| --- | --- | --- |
| 20,000 | about 5 minutes | about 205MB of JPEGs |
| 1,000,000 | about 4 hours | about 10GB of JPEGs, about 20GB once shards are packed |

Leave the workbench at 2 CPU and 8Gi unless you have measured a larger size on your cluster. The log prints a line every 500 files.

```bash
mkdir -p /opt/app-root/src/data/logs
python -u generate_images.py \
  --count 1000000 \
  --workers 2 \
  --output /opt/app-root/src/data/raw \
  | tee /opt/app-root/src/data/logs/generate-million.log
```

The files land here, 250,000 JPEGs per class:

```text
/opt/app-root/src/data/raw/truck/000000.jpg
/opt/app-root/src/data/raw/boat/000000.jpg
/opt/app-root/src/data/raw/plane/000000.jpg
/opt/app-root/src/data/raw/building/000000.jpg
/opt/app-root/src/data/raw/classes.txt
/opt/app-root/src/data/raw/generator.txt
```

Pipeline pods see that directory as `/data/raw`. The stamp in `generator.txt` is `overhead-silhouettes-v1`. A second run with the same stamp fills gaps and leaves existing files. The command stops if you ask for fewer images than the folder already holds. `--replace` deletes the extras.

`run-million.sh` is the unattended form of the draw step plus a pipeline run. It writes the same `/data/raw` tree, then starts `pipeline.py` with the defaults: every image, one epoch, shards under `/data/shards`. The log is `/opt/app-root/src/data/logs/million-run.log`. The ten-epoch numbers below are a separate command.

## Train for ten epochs

One epoch is one pass over the training shards. The model is created once. The next epoch reads the shards again and keeps the weights. The pipeline default is one epoch so the 20,000-image demo finishes quickly.

Before a ten-epoch million fit, set `TIMEOUT_SEC` in `submit_trainjob.py` to at least `10800` and copy that file to `/data/code`. The default wait is 3,600 seconds. The measured fit ran about 75 minutes, which is longer than that wait once pod startup is included.

```bash
python pipeline.py --epochs 10
```

`--epochs 10` with no other flags uses every JPEG in `/data/raw` and writes:

| Path | What |
| --- | --- |
| `/data/splits/split.json` | 800,000 train records and 200,000 test records |
| `/data/shards/train` | 400 shards |
| `/data/shards/test` | 100 shards |
| `/data/shards/manifest.json` | Class list and shard lists |
| `/data/metrics/train-metrics.json` | Epoch history |
| `/data/metrics/test-metrics.json` | Held-out scores and the confusion matrix |
| `/data/models/resnet18.pt` | Checkpoint |

Pack deletes `shard-*.tar` inside those train and test directories before it writes. Point a smaller trial at another shard directory if those shards have to stay.

## Time on a two-GPU sandbox

These times are from one OpenShift AI 3.5 sandbox: workbench at 2 CPU and 8Gi, TrainJob at 2 nodes and 1 GPU each, shards on the RWX volume. Your storage will move the pack time.

| Stage | What it does | Time |
| --- | --- | --- |
| Draw | 1,000,000 JPEGs at about 70 images/sec | about 4 hours |
| Split | Walk the JPEGs and write the 80/20 list | about 2 minutes |
| Pack | Read each JPEG into 2,000-image shards | 2 hours 15 minutes |
| Train | 10 epochs, 800,000 images each, about 1,800 images/sec | 75 minutes |
| Evaluate | 200,000 held-out images | about 2 minutes |

Drawing plus the pipeline is most of a work day, or an overnight chain if you use `run-million.sh` and accept its one-epoch default. Pack is the long pipeline step, and it does not request a GPU. Training is the step that should show both GPUs busy.

On that run, epoch 1 reached about 0.99 train accuracy and later epochs reached 1.0. The held-out test set also scored 1.0. Train and test paths did not overlap: 200,000 and 50,000 images per class. The drawings are easy. More epochs on this generator stay at 1.0. A one-epoch pass on 20,000 images, for comparison, landed near 0.77 train accuracy and 0.83 test accuracy.

GPU utilization during the 75-minute fit, sampled once a minute from the cluster, had medians of about 91 percent and 86 percent. Memory stayed near 7,042 MB on both cards. The rate matched the 20,000-image fit. That is the comparison: a million files in shards did not slow training down. The cost of a million separate files showed up in the pack step.

`images_per_sec` is every image processed during the fit, divided by the wall time. The train step logs one MLflow point per epoch for loss, accuracy, and macro and per-class precision, recall, and F1. The evaluate step logs the confusion matrix. Training pods do not call MLflow. Turn on **Allow pipeline tasks to log directly to MLflow** or the epoch curve, the matrix image, and the GPU series are missing.

## A 100,000-image trial

Draw the million first, or draw at least 25,000 of each class. `--max-per-class 25000` keeps the first 25,000 JPEGs of each class, 100,000 images in total. The trial writes its own shards, metrics, and checkpoint so pack does not remove `/data/shards`.

```bash
python pipeline.py \
  --max-per-class 25000 \
  --epochs 3 \
  --shard-dir /data/subset-100k/shards \
  --metrics-dir /data/subset-100k/metrics \
  --checkpoint /data/subset-100k/models/resnet18.pt
```

Twenty thousand images is `--max-per-class 5000` and a shard directory such as `/data/subset-20k/shards`.

## Leave these alone unless you are changing the demo

- 2,000 images per shard. One million images then split evenly across two GPUs: 400 train shards and 100 test shards.
- Two TrainJob nodes, shard shuffle off, `split_by_node` doing the split.
- Pipeline name `overhead-shard-training`. The graph in this repo is version `v2`. The next graph edit needs a new version name.
- Shared volume of at least 100Gi for the million-image set.

## Later

- Register the checkpoint only after a threshold on `test_accuracy`.
- Put the project under Kueue when the GPUs are shared with other users.
- Mixed precision, if a profile says the fit is compute bound.
