import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segmentation.models import AttentionUNet, UNet, build_model


class ModelShapeTests(unittest.TestCase):
    def test_unet_forward_preserves_spatial_shape(self) -> None:
        model = UNet(in_channels=3, out_channels=1, base_channels=8)
        output = model(torch.randn(2, 3, 128, 128))

        self.assertEqual(tuple(output.shape), (2, 1, 128, 128))

    def test_attention_unet_forward_preserves_spatial_shape(self) -> None:
        model = AttentionUNet(in_channels=3, out_channels=1, base_channels=8)
        output = model(torch.randn(2, 3, 128, 128))

        self.assertEqual(tuple(output.shape), (2, 1, 128, 128))

    def test_build_model_rejects_unknown_architecture(self) -> None:
        with self.assertRaises(ValueError):
            build_model("unknown")


if __name__ == "__main__":
    unittest.main()
