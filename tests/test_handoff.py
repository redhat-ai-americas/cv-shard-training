"""Checks that do not need a GPU, a cluster, or the training image."""

import json
import tempfile
import unittest
from pathlib import Path

import gpu_sample
import submit_trainjob
from generate_images import generate


class GpuMetricsTest(unittest.TestCase):
    def test_parse_nvidia_smi_line(self):
        row = gpu_sample.parse_line("0, 87, 4096", 1000)
        self.assertEqual(row["gpu"], 0)
        self.assertEqual(row["utilization_percentage"], 87.0)
        self.assertEqual(row["memory_usage_megabytes"], round(4096 * 1024 * 1024 / 1e6, 1))
        self.assertIsNone(gpu_sample.parse_line("0, [N/A], 10", 1000))

    def test_metric_names_match_mlflow_system_metrics(self):
        root = Path(tempfile.mkdtemp())
        (root / "gpu-train-node-0.jsonl").write_text(
            json.dumps(
                {
                    "timestamp": 111,
                    "gpu": 0,
                    "utilization_percentage": 99.0,
                    "memory_usage_megabytes": 7043.3,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (root / "gpu-eval-node-0.jsonl").write_text(
            json.dumps(
                {
                    "timestamp": 222,
                    "gpu": 0,
                    "utilization_percentage": 1.0,
                    "memory_usage_megabytes": 1.0,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        rows = submit_trainjob.gpu_metric_batch(root, "train")
        self.assertEqual(
            [row[0] for row in rows],
            [
                "system/node_0/gpu_0_utilization_percentage",
                "system/node_0/gpu_0_memory_usage_megabytes",
            ],
        )
        self.assertEqual(submit_trainjob.gpu_metric_batch(root, "eval")[0][1], 1.0)


class GenerateGuardTest(unittest.TestCase):
    def test_smaller_count_does_not_delete_a_larger_corpus(self):
        root = Path(tempfile.mkdtemp())
        folder = root / "truck"
        folder.mkdir()
        for index in range(3):
            (folder / f"{index:06d}.jpg").write_bytes(b"jpeg")
        with self.assertRaises(SystemExit):
            generate(1, root)
        self.assertEqual(len(list(folder.glob("*.jpg"))), 3)


if __name__ == "__main__":
    unittest.main()
