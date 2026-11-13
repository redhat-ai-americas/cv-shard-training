#!/bin/bash
# Regenerates the million-image corpus under /data/raw, then runs the pipeline
# defaults: every image, one epoch, shards rewritten under /data/shards.
# Pack deletes shard-*.tar inside that directory's train/ and test/ folders.
# The finished ten-epoch fit is documented in scaling-to-one-million.md.
set -euo pipefail
mkdir -p /opt/app-root/src/data/logs
exec >> /opt/app-root/src/data/logs/million-run.log 2>&1
echo "start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
cd /opt/app-root/src
python -u generate_images.py --count 1000000 --workers 2 --output /opt/app-root/src/data/raw
python -u pipeline.py
echo "done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
