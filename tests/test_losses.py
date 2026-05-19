import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segmentation.losses import dice_coefficient, iou_score


class MetricTests(unittest.TestCase):
    def test_dice_and_iou_are_one_for_perfect_predictions(self) -> None:
        logits = torch.full((1, 1, 8, 8), -8.0)
        logits[:, :, 2:6, 2:6] = 8.0
        target = torch.zeros((1, 1, 8, 8))
        target[:, :, 2:6, 2:6] = 1.0

        self.assertAlmostEqual(float(dice_coefficient(logits, target)), 1.0, places=5)
        self.assertAlmostEqual(float(iou_score(logits, target)), 1.0, places=5)

    def test_metrics_handle_empty_prediction_and_target(self) -> None:
        logits = torch.full((2, 1, 8, 8), -8.0)
        target = torch.zeros((2, 1, 8, 8))

        self.assertAlmostEqual(float(dice_coefficient(logits, target)), 1.0, places=5)
        self.assertAlmostEqual(float(iou_score(logits, target)), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
