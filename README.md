# Fixing Ruler-Like Border Artifact Bias in Vision Transformers

This project compares scratch-trained transformer classifiers on the ISIC 2019 skin-lesion dataset and evaluates ruler-like border artifact mitigation strategies.

- `baseline`: `vit_base_patch16_224`
- `proposed model`: `swin_tiny_patch4_window7_224`
- `foundation`: `openai/clip-vit-large-patch14`, zero-shot inference only

## Dataset

Dataset: ISIC 2019 Kaggle dataset  
https://www.kaggle.com/datasets/andrewmvd/isic-2019

The project uses the 8 diagnostic classes `MEL`, `NV`, `BCC`, `AK`, `BKL`, `DF`, `VASC`, and `SCC`. The `UNK` class is excluded.

This is an Option C CSV-split setup. Generate split files with:

```powershell
python dataset_setup.py
```

Expected dataset layout:

```text
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

`dataset_setup.py` uses KaggleHub to download the dataset automatically if `data/isic2019/` is missing.

## Repository Structure

```text
ruler-bias-vit/
  train.py
  test.py
  dataset_setup.py
  requirements.txt
  README.txt
  README.md
  src/
    dataloader.py
    utils.py
    models/
      baseline_vit.py
      swin_transformer.py
      foundation_clip.py
    techniques/
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
      baseline/
      swin_no_technique/
      swin_technique_1/
      swin_technique_2/
      swin_technique_3/
    baseline_none_best/
    foundation/
    swin_none_best/
    swin_technique1_best/
    swin_technique2_best/
    swin_technique3_best/
```

Each evaluated run folder under `outputs/` contains saved metrics and plots. Trained model folders also contain Grad-CAM images and epoch-trend plots. Swin robustness runs include `robustness_comparison.csv`, `test_no_ruler/`, and `test_with_ruler/`.

## Requirements

- Python 3.10+
- Install dependencies:

```powershell
pip install -r requirements.txt
```

The requirements file installs a CUDA-enabled PyTorch build when available.

## Checkpoints

Best checkpoints are included in `models/`:

```text
models/
  baseline_none_best.pth
  swin_none_best.pth
  swin_technique1_best.pth
  swin_technique2_best.pth
  swin_technique3_best.pth
```

Foundation uses zero-shot CLIP inference and does not use a project checkpoint.

## Training

Default training command:

```powershell
python train.py
```

This trains `swin` with `--technique none` from scratch. `baseline` and `swin` always use `pretrained=False`.

Full default Swin training command:

```powershell
python train.py --model swin --technique none --data_root data/isic2019 --splits_dir data/isic2019/splits --epochs 50 --batch_size 64 --lr 1e-4 --device cuda:0 --out_dir outputs/
```

Other training commands:

```powershell
python train.py --model baseline --data_root data/isic2019 --splits_dir data/isic2019/splits --epochs 50 --batch_size 64 --lr 1e-4 --device cuda:0 --out_dir outputs/
python train.py --model swin --technique technique1 --data_root data/isic2019 --splits_dir data/isic2019/splits --epochs 50 --batch_size 64 --lr 1e-4 --device cuda:0 --out_dir outputs/
python train.py --model swin --technique technique2 --data_root data/isic2019 --splits_dir data/isic2019/splits --epochs 50 --batch_size 64 --lr 1e-4 --device cuda:0 --out_dir outputs/
python train.py --model swin --technique technique3 --data_root data/isic2019 --splits_dir data/isic2019/splits --epochs 50 --batch_size 64 --lr 1e-4 --device cuda:0 --out_dir outputs/
```

## Testing

Default test command:

```powershell
python test.py
```

This evaluates `swin` with `models/swin_none_best.pth`.

Full default Swin test command:

```powershell
python test.py --model swin --ckpt models/swin_none_best.pth --data_root data/isic2019 --splits_dir data/isic2019/splits --batch_size 64 --device cuda:0
```

Other test commands:

```powershell
python test.py --model baseline --ckpt models/baseline_none_best.pth --data_root data/isic2019 --splits_dir data/isic2019/splits --batch_size 64 --device cuda:0
python test.py --model foundation --data_root data/isic2019 --splits_dir data/isic2019/splits --batch_size 64 --device cuda:0
python test.py --model swin --ckpt models/swin_none_best.pth --robustness_test --data_root data/isic2019 --splits_dir data/isic2019/splits --batch_size 64 --device cuda:0
python test.py --model swin --ckpt models/swin_technique1_best.pth --robustness_test --data_root data/isic2019 --splits_dir data/isic2019/splits --batch_size 64 --device cuda:0
python test.py --model swin --ckpt models/swin_technique2_best.pth --robustness_test --data_root data/isic2019 --splits_dir data/isic2019/splits --batch_size 64 --device cuda:0
python test.py --model swin --ckpt models/swin_technique3_best.pth --robustness_test --data_root data/isic2019 --splits_dir data/isic2019/splits --batch_size 64 --device cuda:0
```

## Sanity Metrics

Expected approximate test metrics:

- Default Swin no-technique checkpoint: macro recall `0.5725`, macro F1 `0.4342`
- Best Swin Technique 1 checkpoint: macro recall `0.5960`, macro F1 `0.4471`

## Notes

- `baseline` and `swin` train from scratch only.
- `foundation` is test-only and runs zero-shot CLIP inference with pretrained CLIP weights and no project checkpoint.
- The first `python test.py --model foundation` run downloads the pretrained CLIP model/tokenizer from Hugging Face unless already cached.
- ISIC 2019 does not include explicit ruler annotations in this setup, so `test_no_ruler.csv` and `test_with_ruler.csv` are duplicated from the same test split and should be interpreted as a pipeline sanity check.
