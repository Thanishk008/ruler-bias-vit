# Ruler Bias in Skin Lesion Classification

This project compares three image-classification models on the ISIC 2019 skin-lesion dataset while studying ruler-bias effects:

- `baseline`: ViT baseline
- `swin`: Swin Transformer
- `foundation`: frozen CLIP linear probe

## Dataset

The repo uses the ISIC 2019 Kaggle dataset:

https://www.kaggle.com/datasets/andrewmvd/isic-2019

Only the 8 real diagnostic classes are used. The `UNK` class is excluded.

Expected layout:

```text
data/
  isic2019/
    ISIC_2019_Training_Input/
      ISIC_2019_Training_Input/
        <image_id>.jpg
    ISIC_2019_Training_GroundTruth.csv
    ISIC_2019_Training_Metadata.csv
    splits/
```

Generate the split CSVs with:

```powershell
python dataset_setup.py
```

`dataset_setup.py` uses KaggleHub to download the dataset automatically if `data/isic2019/` is missing.
The Kaggle archive nests `ISIC_2019_Training_Input` one level deep, so the raw download keeps that folder structure.

## Requirements

- Python 3.10+
- Install dependencies:

```powershell
pip install -r requirements.txt
```

## Training

Default training command:

```powershell
python train.py
```

Defaults:

- model: `swin`
- technique: `none`
- pretrained: `True`

Other useful commands:

```powershell
python train.py --model baseline
python train.py --model swin --technique technique1
python train.py --model swin --technique technique2
python train.py --model swin --technique technique3
python train.py --model foundation
python train_all.py
```

## Testing

Default test command:

```powershell
python test.py
```

Defaults:

- model: `swin`
- checkpoint: `models/swin_none_best.pth`

Example test commands:

```powershell
python test.py --model baseline --ckpt models/baseline_none_best.pth
python test.py --model foundation --ckpt models/foundation_none_best.pth
python test.py --model swin --ckpt models/swin_none_best.pth --robustness_test
python test.py --model swin --ckpt models/swin_technique1_best.pth --robustness_test
python test.py --model swin --ckpt models/swin_technique2_best.pth --robustness_test
python test.py --model swin --ckpt models/swin_technique3_best.pth --robustness_test
```

## Outputs

Training checkpoints are written under `models/`.

Testing writes metrics and plots under `outputs/<model>/`, including:

- `metrics.csv`
- `confusion_matrix.png`
- `pr_curve.png`
- `gradcam/`

If `--robustness_test` is set, the test script also writes:

- `outputs/<model>/robustness_comparison.csv`
- `outputs/<model>/test_no_ruler/`
- `outputs/<model>/test_with_ruler/`

## Notes

- `baseline` and `swin` fine-tune pretrained weights.
- `foundation` keeps the CLIP backbone frozen and trains only the classifier head.
- The codebase is set up for Python 3.10+ because the current KaggleHub dependency requires it.
