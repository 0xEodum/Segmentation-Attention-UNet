# Brain MRI Segmentation

This project trains a plain U-Net baseline and an Attention U-Net on the `kaggle_3m` TCGA brain MRI segmentation dataset.

## Train

Install the package first:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

Run commands from the repository root with the project virtual environment:

```powershell
segmentation-train --architecture unet --epochs 30 --batch-size 32 --num-workers 6
segmentation-train --architecture attention_unet --epochs 30 --batch-size 32 --num-workers 6 --checkpoint-every 5
```

Each run writes:

- `runs/<timestamp>_<architecture>/best.pt`
- `runs/<timestamp>_<architecture>/last.pt`
- `runs/<timestamp>_<architecture>/summary.json`
- `checkpoints/<architecture>_best.pt`

The split is patient-folder based, so slices from one `TCGA_*` folder do not cross train/validation/test boundaries.

## Colab

The notebook [Segmentation_Attention_UNet_Colab.ipynb](Segmentation_Attention_UNet_Colab.ipynb) installs this package from GitHub:

```python
%pip install -q "segmentation-attention-unet[notebook] @ git+https://github.com/0xEodum/Segmentation-Attention-UNet.git"
```

It downloads the dataset archive, selects the top-level `kaggle_3m/` directory when both identical archive copies are present, trains both models, and visualizes Attention U-Net gate maps from saved epoch checkpoints.

## Inference

```powershell
segmentation-infer `
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
