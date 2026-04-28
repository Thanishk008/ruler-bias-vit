# README Template
------------------------------------------------------------

## 1. Project Overview

Project Title: Fixing Ruler Bias in Vision Transformers for Skin Lesion Classification

Model Type:
baseline - vit_base_patch16_224
my model - swin_tiny_patch4_window7_224
foundation - openai/clip-vit-large-patch14 (zero-shot inference)

Objective:
Multi-class classification

Dataset Used:
ISIC 2019 Kaggle dataset: https://www.kaggle.com/datasets/andrewmvd/isic-2019

Classes used:
8 diagnostic classes from ISIC 2019
The UNK class is excluded.

Expected test evaluation for sanity check: 
The test script should run end-to-end, produce finite metrics, and save the expected output files; 
exact values vary by checkpoint, split, and training time.

------------------------------------------------------------

## 2. Repository Structure

List the structure of your project directory below. Add short descriptions if needed.

```
ruler-bias-vit/
  train.py
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
  data/                   (dataset location)
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
      ISIC_2019_Training_Input/
        <image_id>.jpg
    ISIC_2019_Training_GroundTruth.csv
    ISIC_2019_Training_Metadata.csv
    splits/
```

`dataset_setup.py` will download the dataset with KaggleHub if `data/isic2019/` is missing. 
This setup expects the latest KaggleHub release, which requires Python 3.10+.
You can also download the dataset manually from Kaggle and place it there first, then generate the split CSVs:

Run the bewlo command only after activating the virtual environment. (Instructions are further down)
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
{}

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
The outputs are written to a run-specific folder under `outputs/`, derived from the checkpoint name for baseline and Swin runs.

------------------------------------------------------------

## 7. Running the Training Script

Default training command:
```
python train.py
```

This trains `swin` from scratch; 
`baseline` also trains from scratch.

`foundation` is test-only in this setup and runs zero-shot CLIP inference with pretrained CLIP weights and no checkpoint.
The first `python test.py --model foundation` run will download the pretrained CLIP model/tokenizer from Hugging Face unless they are already cached locally.

Exact commands after loading the dataset:
```
python train.py --model baseline
python train.py --model swin --technique none
python train.py --model swin --technique technique1
python train.py --model swin --technique technique2
python train.py --model swin --technique technique3

python test.py --model baseline --ckpt models/baseline_none_best.pth
python test.py --model foundation
python test.py --model swin --ckpt models/swin_none_best.pth --robustness_test
python test.py --model swin --ckpt models/swin_technique1_best.pth --robustness_test
python test.py --model swin --ckpt models/swin_technique2_best.pth --robustness_test
python test.py --model swin --ckpt models/swin_technique3_best.pth --robustness_test
```

------------------------------------------------------------

## 8. Submission Checklist

- [ ] Dataset downloaded and split CSVs created in `data/isic2019/splits/`.
- [ ] Model checkpoint linked and instructions for placement included.
- [ ] `requirements.txt` generated and Python version specified.
- [ ] `python test.py` works for the default Swin checkpoint.
- [ ] `python train.py` works for the default Swin run.

------------------------------------------------------------
