# README
------------------------------------------------------------

## 1. Project Overview

Project Title:
Fixing Ruler-Like Border Artifact Bias in Vision Transformers for Skin Lesion Classification

Model Type:
baseline - vit_base_patch16_224
proposed model - swin_tiny_patch4_window7_224
foundation - openai/clip-vit-large-patch14 (zero-shot inference)

Objective:
Multi-class skin lesion classification.

Dataset Used:
ISIC 2019 Kaggle dataset: https://www.kaggle.com/datasets/andrewmvd/isic-2019

Classes used:
8 diagnostic classes from ISIC 2019: MEL, NV, BCC, AK, BKL, DF, VASC, SCC.
The UNK class is excluded.

Expected test evaluation for sanity check:
Default Swin no-technique checkpoint:
Macro Recall approximately 0.5725, Macro F1 approximately 0.4342.

Best Swin Technique 1 checkpoint:
Macro Recall approximately 0.5960, Macro F1 approximately 0.4471.

------------------------------------------------------------

## 2. Repository Structure

```
ruler-bias-vit/
  train.py
  test.py
  dataset_setup.py
  requirements.txt
  README.txt
  README.md

  src/
    __init__.py
    dataloader.py
    utils.py
    models/
      __init__.py
      baseline_vit.py
      swin_transformer.py
      foundation_clip.py
    techniques/
      __init__.py
      technique1_debiasing.py
      technique2_attention_reg.py
      technique3_patch_masking.py

  models/
    baseline_none_best.pth
    swin_none_best.pth
    swin_technique1_best.pth
    swin_technique2_best.pth
    swin_technique3_best.pth

  outputs/
    checkpoints/
      baseline_none_epoch_10.pth
      ...
      swin_technique3_epoch_50.pth

    baseline_none_best/
      metrics.csv
      confusion_matrix.png
      pr_curve.png
      gradcam/
      epoch_trend/

    swin_none_best/
      metrics.csv
      confusion_matrix.png
      pr_curve.png
      robustness_comparison.csv
      gradcam/
      epoch_trend/
      test_no_ruler/
      test_with_ruler/

    swin_technique1_best/
      metrics.csv
      confusion_matrix.png
      pr_curve.png
      robustness_comparison.csv
      gradcam/
      epoch_trend/
      test_no_ruler/
      test_with_ruler/

    swin_technique2_best/
      metrics.csv
      confusion_matrix.png
      pr_curve.png
      robustness_comparison.csv
      gradcam/
      epoch_trend/
      test_no_ruler/
      test_with_ruler/

    swin_technique3_best/
      metrics.csv
      confusion_matrix.png
      pr_curve.png
      robustness_comparison.csv
      gradcam/
      epoch_trend/
      test_no_ruler/
      test_with_ruler/

    foundation/
      metrics.csv
      confusion_matrix.png
      pr_curve.png
      gradcam/              (empty; Grad-CAM is skipped for zero-shot CLIP)

  data/
    isic2019/
      ISIC_2019_Training_Input/
        ISIC_2019_Training_Input/
          <image_id>.jpg
      ISIC_2019_Training_GroundTruth.csv
      ISIC_2019_Training_Metadata.csv
      splits/
        train.csv
        val.csv
        test.csv
        test_no_ruler.csv
        test_with_ruler.csv
```

------------------------------------------------------------

## 3. Dataset Setup - Option C: CSV Split File

Dataset Link:
https://www.kaggle.com/datasets/andrewmvd/isic-2019

Path to generated CSV split files:
```
data/isic2019/splits/
```

CSV format:
```
image_path,label,split
ISIC_2019_Training_Input/ISIC_2019_Training_Input/<image_id>.jpg,0,train
ISIC_2019_Training_Input/ISIC_2019_Training_Input/<image_id>.jpg,1,test
```

Expected dataset layout after download:
```
data/
  isic2019/
    ISIC_2019_Training_Input/
      ISIC_2019_Training_Input/
        <image_id>.jpg
    ISIC_2019_Training_GroundTruth.csv
    ISIC_2019_Training_Metadata.csv
    splits/
```

`dataset_setup.py` will download the dataset with KaggleHub if `data/isic2019/` is missing.
This setup expects Python 3.10+.
You can also download the dataset manually from Kaggle and place it there first.

Run the below command after activating the virtual environment:
```
python dataset_setup.py
```

Notes:
- `dataset_setup.py` excludes `UNK` and uses the 8 diagnostic classes.
- The script creates stratified train/validation/test CSV splits.
- ISIC 2019 does not include explicit ruler annotations in this setup, so `test_no_ruler.csv` and `test_with_ruler.csv` are duplicated from the same test split and should be interpreted only as a pipeline sanity check.

------------------------------------------------------------

## 4. Model Checkpoint

Box Link to Best Model Checkpoints:
TODO: add Box link before submission.

Give access to:
yusun@usf.edu, kandiyana@usf.edu

Where to place checkpoints:
```
models/
  baseline_none_best.pth
  swin_none_best.pth
  swin_technique1_best.pth
  swin_technique2_best.pth
  swin_technique3_best.pth
```

Foundation uses zero-shot CLIP inference and does not use a project checkpoint.

------------------------------------------------------------

## 5. Requirements (Dependencies)

Python Version:
3.10+

Install dependencies:
```
pip install -r requirements.txt
```

Virtual environment setup:
```
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

The requirements file installs a CUDA-enabled PyTorch build when available.

------------------------------------------------------------

## 6. Running the Test Script

Default test command:
```
python test.py
```

This evaluates `swin` with `models/swin_none_best.pth` unless `--model` and `--ckpt` are provided.
Outputs are written to a run-specific folder under `outputs/`, derived from the checkpoint name for baseline and Swin runs.

Full default Swin test command:
```
python test.py --model swin --ckpt models/swin_none_best.pth --data_root data/isic2019 --splits_dir data/isic2019/splits --batch_size 64 --device cuda:0
```

Other test commands:
```
python test.py --model baseline --ckpt models/baseline_none_best.pth --data_root data/isic2019 --splits_dir data/isic2019/splits --batch_size 64 --device cuda:0
python test.py --model foundation --data_root data/isic2019 --splits_dir data/isic2019/splits --batch_size 64 --device cuda:0
python test.py --model swin --ckpt models/swin_none_best.pth --robustness_test --data_root data/isic2019 --splits_dir data/isic2019/splits --batch_size 64 --device cuda:0
python test.py --model swin --ckpt models/swin_technique1_best.pth --robustness_test --data_root data/isic2019 --splits_dir data/isic2019/splits --batch_size 64 --device cuda:0
python test.py --model swin --ckpt models/swin_technique2_best.pth --robustness_test --data_root data/isic2019 --splits_dir data/isic2019/splits --batch_size 64 --device cuda:0
python test.py --model swin --ckpt models/swin_technique3_best.pth --robustness_test --data_root data/isic2019 --splits_dir data/isic2019/splits --batch_size 64 --device cuda:0
```

------------------------------------------------------------

## 7. Running the Training Script

Default training command:
```
python train.py
```

This trains `swin` with `--technique none` from scratch.
`baseline` and `swin` are always trained from scratch with `pretrained=False`.

Full default Swin training command:
```
python train.py --model swin --technique none --data_root data/isic2019 --splits_dir data/isic2019/splits --epochs 50 --batch_size 64 --lr 1e-4 --device cuda:0 --out_dir outputs/
```

Other training commands:
```
python train.py --model baseline --data_root data/isic2019 --splits_dir data/isic2019/splits --epochs 50 --batch_size 64 --lr 1e-4 --device cuda:0 --out_dir outputs/
python train.py --model swin --technique technique1 --data_root data/isic2019 --splits_dir data/isic2019/splits --epochs 50 --batch_size 64 --lr 1e-4 --device cuda:0 --out_dir outputs/
python train.py --model swin --technique technique2 --data_root data/isic2019 --splits_dir data/isic2019/splits --epochs 50 --batch_size 64 --lr 1e-4 --device cuda:0 --out_dir outputs/
python train.py --model swin --technique technique3 --data_root data/isic2019 --splits_dir data/isic2019/splits --epochs 50 --batch_size 64 --lr 1e-4 --device cuda:0 --out_dir outputs/
```

Optional training arguments:
- `--resume`
- `--save_every`
- `--seed`
- `--num_workers`
- `--amp`

Foundation is test-only in this setup and runs zero-shot CLIP inference with pretrained CLIP weights and no project checkpoint.
The first `python test.py --model foundation` run will download the pretrained CLIP model/tokenizer from Hugging Face unless they are already cached locally.

------------------------------------------------------------

## 8. Submission Checklist

- [ ] Dataset downloaded and split CSVs created in `data/isic2019/splits/`.
- [ ] Best model checkpoints are included in `models/` or provided through a shared Box link.
- [ ] `requirements.txt` and Python version are specified.
- [ ] `python test.py` works for the default Swin checkpoint.
- [ ] `python train.py` works for the default Swin run.
- [ ] Foundation test runs without a project checkpoint.

------------------------------------------------------------
