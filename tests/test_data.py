import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segmentation.data import BrainTumorDataset, find_image_mask_pairs, split_by_patient


def _write_pair(root: Path, patient: str, index: int, mask_value: int) -> None:
    folder = root / patient
    folder.mkdir(parents=True, exist_ok=True)
    image = np.full((16, 16, 3), 32 + index, dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:8, 5:9] = mask_value
    stem = f"{patient}_{index}"
    Image.fromarray(image).save(folder / f"{stem}.tif")
    Image.fromarray(mask).save(folder / f"{stem}_mask.tif")


class BrainTumorDataTests(unittest.TestCase):
    def test_finds_pairs_without_treating_masks_as_images(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_pair(root, "TCGA_A", 1, 255)
            _write_pair(root, "TCGA_A", 2, 0)

            pairs = find_image_mask_pairs(root)

            self.assertEqual(len(pairs), 2)
            self.assertTrue(all(not pair.image_path.name.endswith("_mask.tif") for pair in pairs))
            self.assertTrue(all(pair.mask_path.exists() for pair in pairs))

    def test_patient_split_has_no_folder_leakage(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for patient_idx in range(10):
                for image_idx in range(2):
                    _write_pair(root, f"TCGA_{patient_idx:02d}", image_idx, 255)

            pairs = find_image_mask_pairs(root)
            split = split_by_patient(pairs, val_fraction=0.2, test_fraction=0.2, seed=7)

            patient_sets = {
                name: {pair.patient_id for pair in values}
                for name, values in split.items()
            }
            self.assertTrue(patient_sets["train"].isdisjoint(patient_sets["val"]))
            self.assertTrue(patient_sets["train"].isdisjoint(patient_sets["test"]))
            self.assertTrue(patient_sets["val"].isdisjoint(patient_sets["test"]))
            self.assertEqual(sum(len(values) for values in split.values()), len(pairs))

    def test_dataset_returns_normalized_image_and_binary_mask(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_pair(root, "TCGA_A", 1, 255)
            pair = find_image_mask_pairs(root)[0]
            dataset = BrainTumorDataset([pair], augment=False)

            image, mask, meta = dataset[0]

            self.assertEqual(tuple(image.shape), (3, 256, 256))
            self.assertEqual(tuple(mask.shape), (1, 256, 256))
            self.assertGreaterEqual(float(image.min()), 0.0)
            self.assertLessEqual(float(image.max()), 1.0)
            self.assertEqual(set(mask.unique().tolist()), {0.0, 1.0})
            self.assertEqual(meta["patient_id"], "TCGA_A")


if __name__ == "__main__":
    unittest.main()
