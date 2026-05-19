# Brain MRI Segmentation

This project trains a plain U-Net baseline and an Attention U-Net on the `kaggle_3m` TCGA brain MRI segmentation dataset.

## Train

Run commands from the repository root with the project virtual environment:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m segmentation.train --architecture unet --epochs 30 --batch-size 32 --num-workers 6
.\.venv\Scripts\python.exe -m segmentation.train --architecture attention_unet --epochs 30 --batch-size 32 --num-workers 6
```

Each run writes:

- `runs/<timestamp>_<architecture>/best.pt`
- `runs/<timestamp>_<architecture>/last.pt`
- `runs/<timestamp>_<architecture>/summary.json`
- `checkpoints/<architecture>_best.pt`

The split is patient-folder based, so slices from one `TCGA_*` folder do not cross train/validation/test boundaries.

## Inference

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m segmentation.infer `
  --checkpoint checkpoints\attention_unet_best.pt `
  --input kaggle_3m\TCGA_CS_4941_19960909\TCGA_CS_4941_19960909_1.tif `
  --output-dir predictions\attention_unet `
  --save-overlay
```

The CLI accepts either one image or a directory. It writes predicted mask PNG files and a `predictions.json` summary.

## Test

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m unittest discover -s tests
```
