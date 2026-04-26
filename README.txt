# README Template
------------------------------------------------------------

## 1. Project Overview

Project Title: Fixing Ruler Bias in Vision Transformers for Skin Lesion Classification

Model Type:
Vision Transformer + Swin Transformer + CLIP linear probe

Objective:
Multi-class classification

Dataset Used:
ISIC 2019 Kaggle dataset: https://www.kaggle.com/datasets/andrewmvd/isic-2019

Classes used:
8 diagnostic classes from ISIC 2019
The UNK class is excluded.

Expected test evaluation for sanity check: Macro F1 varies by checkpoint and split; a quick CPU sanity run on Swin for 1 epoch should complete successfully.

------------------------------------------------------------

## 2. Repository Structure

List the structure of your project directory below. Add short descriptions if needed.

Example (replace with your own):
```
ruler-bias-vit/
  train.py
  train_all.py
  test.py
  requirements.txt
  dataset_setup.py
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
  models/                 (checkpoint goes here)
  outputs/                (evaluation outputs and plots)
  README.txt
```

------------------------------------------------------------

## 3. Dataset Setup

Dataset Link:
https://www.kaggle.com/datasets/andrewmvd/isic-2019

Expected layout after download:
```
data/
  isic2019/
    ISIC_2019_Training_Input/
      <image_id>.jpg
    ISIC_2019_Training_GroundTruth.csv
    ISIC_2019_Training_Metadata.csv
    splits/
```

`dataset_setup.py` will download the dataset with KaggleHub if `data/isic2019/` is missing. This setup expects the latest KaggleHub release, which requires Python 3.10+, so recreate the venv with Python 3.10+ if you are still on 3.9. You can also download the dataset manually from Kaggle and place it there first, then generate the split CSVs:
```
python dataset_setup.py
```

Notes:
- `dataset_setup.py` excludes `UNK` and uses the 8 real diagnostic classes.
- ISIC 2019 does not include ruler annotations, so `test_no_ruler.csv` and `test_with_ruler.csv` are duplicated from the same test split.
- The script generates the split CSVs and can fetch the dataset automatically if needed.
------------------------------------------------------------

## 4. Model Checkpoint

Box Link to Best Model Checkpoint:
N/A

Give access to:
yusun@usf.edu, kandiyana@usf.edu

Where to place the checkpoint after downloading:
```
models/
  swin_none_best.pth
```

------------------------------------------------------------

## 5. Requirements (Dependencies)

Python Version:
3.10+

How to install all dependencies (e.g. requirements.txt):

Using pip:
```
pip install -r requirements.txt
```

Virtual Env:
```
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

------------------------------------------------------------

## 6. Running the Test Script

Default test command:
```
python test.py
```

This evaluates `swin` with `models/swin_none_best.pth` unless you pass `--model` and `--ckpt`.

------------------------------------------------------------

## 7. Running the Training Script

Default training command:
```
python train.py
```

This fine-tunes `swin` from pretrained weights by default; 
`baseline` also fine-tunes a pretrained ViT, and `foundation` keeps the CLIP backbone frozen.

Train all three models back-to-back:
```
python train_all.py
```

Exact commands after loading the dataset:
```
python train.py --model baseline
python train.py --model swin --technique none
python train.py --model swin --technique technique1
python train.py --model swin --technique technique2
python train.py --model swin --technique technique3
python train.py --model foundation

python test.py --model baseline --ckpt models/baseline_none_best.pth
python test.py --model foundation --ckpt models/foundation_none_best.pth
python test.py --model swin --ckpt models/swin_none_best.pth --robustness_test
python test.py --model swin --ckpt models/swin_technique1_best.pth --robustness_test
python test.py --model swin --ckpt models/swin_technique2_best.pth --robustness_test
python test.py --model swin --ckpt models/swin_technique3_best.pth --robustness_test
```

Optional arguments (if supported):
- `--resume`
- `--save_every`
- `--seed`
- `--pretrained`
- `--num_workers`

------------------------------------------------------------

## 8. Submission Checklist

- [ ] Dataset downloaded and split CSVs created in `data/isic2019/splits/`.
- [ ] Model checkpoint linked and instructions for placement included.
- [ ] `requirements.txt` generated and Python version specified.
- [ ] `python test.py` works for the default Swin checkpoint.
- [ ] `python train.py` works for the default Swin run.

------------------------------------------------------------
