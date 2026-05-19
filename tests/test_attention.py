import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segmentation.attention import collect_attention_maps, forward_with_attention
from segmentation.models import AttentionUNet, UNet


class AttentionCaptureTests(unittest.TestCase):
    def test_forward_with_attention_collects_all_attention_gates(self) -> None:
        model = AttentionUNet(in_channels=3, out_channels=1, base_channels=8)
        images = torch.randn(1, 3, 128, 128)

        logits, maps = forward_with_attention(model, images, upsample_to=(128, 128))

        self.assertEqual(tuple(logits.shape), (1, 1, 128, 128))
        self.assertEqual(set(maps), {"up1.attention", "up2.attention", "up3.attention", "up4.attention"})
        for attention in maps.values():
            self.assertEqual(tuple(attention.shape), (1, 1, 128, 128))
            self.assertFalse(attention.requires_grad)
            self.assertGreaterEqual(float(attention.min()), 0.0)
            self.assertLessEqual(float(attention.max()), 1.0)

    def test_plain_unet_has_no_attention_maps(self) -> None:
        model = UNet(in_channels=3, out_channels=1, base_channels=8)
        images = torch.randn(1, 3, 128, 128)

        _, maps = forward_with_attention(model, images, upsample_to=(128, 128))

        self.assertEqual(maps, {})
        self.assertEqual(collect_attention_maps(model), {})


if __name__ == "__main__":
    unittest.main()
