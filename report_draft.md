# Fixing Ruler-Like Border Artifact Bias in Vision Transformers for Skin Lesion Classification

## 1. Project Goal

The goal of this project is to design, train, and evaluate transformer-based models for multi-class skin lesion classification on the ISIC 2019 dataset, with a specific focus on reducing sensitivity to ruler-like border artifacts. In dermoscopic imaging, rulers, dark borders, and acquisition artifacts can become visual shortcuts that a model may learn instead of clinically meaningful lesion features. This project studies whether targeted border-artifact mitigation strategies can improve a scratch-trained transformer classifier.

The final task is an 8-class classification problem using the ISIC 2019 diagnostic labels: MEL, NV, BCC, AK, BKL, DF, VASC, and SCC. The UNK class is excluded. The project compares a scratch-trained Vision Transformer baseline, a scratch-trained Swin Transformer proposed model, three Swin-based improvement techniques, and a zero-shot CLIP foundation model used only for inference.

The classification goal was achieved: the proposed Swin Transformer trained from scratch outperformed the baseline ViT on macro precision, macro recall, and macro F1. The best-performing improvement method was Technique 1, which combined border-focused preprocessing with synthetic ruler-like augmentation. The ruler-bias goal was addressed through artifact-mitigation strategies and qualitative visualization, but direct ruler robustness could not be measured because the dataset configuration used in this project does not provide explicit ruler/no-ruler annotations. Therefore, the ruler-related evaluation should be interpreted as a study of ruler-like border artifact mitigation rather than a definitive ruler-present versus ruler-absent robustness study.

## 2. Input and Ground Truth

The input to each trained model is a dermoscopic RGB skin-lesion image resized and normalized for transformer classification. The ground truth label is one of eight diagnostic classes from ISIC 2019:

| Label | Class | Meaning |
| ---: | --- | --- |
| 0 | MEL | Melanoma |
| 1 | NV | Melanocytic nevus |
| 2 | BCC | Basal cell carcinoma |
| 3 | AK | Actinic keratosis |
| 4 | BKL | Benign keratosis |
| 5 | DF | Dermatofibroma |
| 6 | VASC | Vascular lesion |
| 7 | SCC | Squamous cell carcinoma |

The model output is an 8-dimensional logit vector, where each logit corresponds to one diagnostic class. The predicted class is the index with the highest softmax probability. The relationship between input and output is therefore standard supervised image classification: each image is mapped to one diagnosis label.

The project excludes the ISIC 2019 UNK label. Rows marked as UNK are removed during dataset setup, and the remaining samples are required to have one positive label among the eight retained classes.

Compared with the proposal-stage framing, the final input and ground-truth representation remained a supervised dermoscopic-image classification task. The main change is in the ruler-bias evaluation: the project originally targeted ruler-related bias, but the final dataset configuration did not include explicit ruler/no-ruler annotations. As a result, the ruler component is framed as ruler-like border-artifact mitigation using targeted interventions, robustness checks, and qualitative attention/error analysis rather than as a direct annotated ruler-bias benchmark.

## 3. Dataset Description

The dataset used in this project is the public ISIC 2019 Kaggle dataset:

https://www.kaggle.com/datasets/andrewmvd/isic-2019

The dataset consists of dermoscopic skin lesion images with diagnostic ground-truth labels and metadata. In this project, the data were prepared by removing UNK samples, encoding the eight retained classes, and creating stratified train/validation/test splits.

### Split Sizes

The dataset was split into an 80/10/10 train/validation/test split using stratified sampling with seed 42.

| Split | Count |
| --- | ---: |
| Train | 20264 |
| Validation | 2533 |
| Test | 2534 |
| Test no-ruler proxy | 2534 |
| Test with-ruler proxy | 2534 |

The no-ruler and with-ruler proxy splits are duplicated from the same test split because explicit ruler/no-ruler annotations are not available in this dataset configuration. They are useful for checking evaluation consistency, but they should not be interpreted as evidence of real ruler-specific robustness.

### Class Distribution and Imbalance

The training split is strongly imbalanced. NV dominates the dataset, while DF, VASC, SCC, and AK are much smaller. This imbalance motivates the use of macro-averaged metrics and class-weighted cross-entropy.

| Label | Class | Train Count | Class Weight |
| ---: | --- | ---: | ---: |
| 0 | MEL | 3618 | 0.7001 |
| 1 | NV | 10300 | 0.2459 |
| 2 | BCC | 2658 | 0.9530 |
| 3 | AK | 694 | 3.6499 |
| 4 | BKL | 2099 | 1.2068 |
| 5 | DF | 191 | 13.2618 |
| 6 | VASC | 202 | 12.5396 |
| 7 | SCC | 502 | 5.0458 |

## 4. Data Processing

Dataset setup performs the following steps:

- Loads the ISIC 2019 ground-truth CSV and image paths.
- Removes rows belonging to the UNK class.
- Keeps the eight diagnostic classes used in this project.
- Converts the one-hot class columns into integer labels.
- Creates stratified train, validation, and test splits with seed 42.
- Stores split assignments as CSV files for reproducible reuse.

The trained ViT and Swin models use ImageNet-style input normalization after resizing/cropping and tensor conversion. The training transform includes data augmentation, while validation and test transforms use deterministic preprocessing. Class weights are computed from the training split and passed to cross-entropy loss to reduce the effect of class imbalance.

The foundation model uses CLIP-specific preprocessing and normalization because zero-shot CLIP expects a different image normalization scheme from standard ImageNet-trained transformer classifiers.

Data leakage is avoided by creating the train, validation, and test splits before model training and by selecting the best checkpoint using validation macro recall, not test performance. The test set is used only for final evaluation. The duplicated no-ruler and with-ruler proxy splits are derived from the same test set and are not used for training.

## 5. Methodology

### 5.1 Baseline Model

The baseline model is a Vision Transformer using the `vit_base_patch16_224` architecture from `timm`. It is prebuilt but not pretrained:

```text
timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=8)
```

This satisfies the project requirement that proposed models be trained from scratch. The baseline is trained on the ISIC 2019 training split with an 8-class classification head. It provides a transformer-based reference point for comparison against the proposed Swin Transformer and improvement techniques.

### 5.2 Proposed Transformer Model Architecture

The proposed model is a Swin Transformer using `swin_tiny_patch4_window7_224` from `timm`, also with `pretrained=False`:

```text
timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=8)
```

Swin Transformer is appropriate for this problem because it uses hierarchical feature maps and shifted-window self-attention. Compared with a standard ViT, which tokenizes the whole image into fixed patches and applies global self-attention across a flat token sequence, Swin builds multi-stage local representations through windowed attention and shifted windows. This design reduces attention cost and encourages local-to-global feature learning. That is useful for dermoscopic images because diagnostic information may appear in local lesion texture, border irregularity, color variation, and spatial structure around the lesion.

The proposed Swin model is evaluated in four trained configurations:

- Swin without additional technique.
- Swin with Technique 1.
- Swin with Technique 2.
- Swin with Technique 3.

### 5.3 Training Details and Reproducibility

Training is implemented in PyTorch. The default training settings are:

| Setting | Value |
| --- | --- |
| Framework | PyTorch |
| Model library | timm |
| Epochs | 50 |
| Batch size | 64 by default |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingLR |
| Loss | Weighted cross-entropy |
| Seed | 42 |
| Checkpoint frequency | Every 10 epochs |
| Best checkpoint criterion | Validation macro recall |
| AMP | Optional, disabled by default |

Best-model and periodic checkpoints were stored throughout training. Training can be resumed from an intermediate checkpoint, and each saved checkpoint stores enough information to restore the model, optimizer, scheduler, epoch, and best validation recall.

The baseline and all Swin models are trained from scratch. No pretrained weights are used for these trained models. The foundation model is different by design: CLIP is pretrained, but it is used only for zero-shot inference and is not trained or fine-tuned in this project.

The experiments were run in a GPU-enabled Google Colab environment. During development, an A100-class Colab GPU was used for the larger Swin runs. The implementation also supports explicit control over device selection, batch size, worker count, mixed precision, save frequency, and checkpoint resumption.

## 6. Performance Improvement Techniques

The project implements and evaluates three improvement techniques, all applied to the Swin Transformer.

### Technique 1: Border/Ruler Preprocessing and Synthetic Ruler Augmentation

Technique 1 modifies the training transform by adding border-focused preprocessing and synthetic ruler-like augmentation. It applies `RulerCropTransform` and `SyntheticRulerAugmentation` before tensor conversion.

The motivation is to reduce the model's ability to rely on border/ruler shortcuts and expose it to controlled ruler-like artifacts during training. This can encourage the model to learn more lesion-relevant features instead of overfitting to acquisition artifacts.

Technique 1 produced the best overall trained result, with the highest macro recall and macro F1 among all trained Swin runs.

### Technique 2: Border-Patch Attention Regularization

Technique 2 adds an attention regularization loss with `lambda_weight=0.1`. The regularizer penalizes attention assigned to predefined border patch indices.

The motivation is to discourage the model from focusing too strongly on border regions, where ruler-like or image-frame artifacts are most likely to appear. This technique improved macro recall relative to Swin without techniques, but its macro F1 was lower than Technique 1.

### Technique 3: Border-Patch Masking

Technique 3 uses a border patch masker with `mask_prob=0.5`. During training, border patch embeddings are masked so that the model cannot consistently depend on border information.

The motivation is similar to dropout or input masking: by hiding potentially biased border features, the model may be forced to rely more on lesion-relevant image content. In practice, this was the weakest Swin configuration. It reduced accuracy and macro F1 relative to the Swin no-technique model, suggesting that the masking may have removed useful context or made optimization harder.

### Technique Impact Summary

The table below compares each technique against the Swin no-technique model. Positive changes indicate improvement relative to Swin without techniques.

| Run | Macro Precision | Macro Recall | Macro F1 | Delta Recall vs. Swin None | Delta F1 vs. Swin None |
| --- | ---: | ---: | ---: | ---: | ---: |
| Swin no technique | 0.4048 | 0.5725 | 0.4342 | 0.0000 | 0.0000 |
| Swin Technique 1 | 0.4104 | 0.5960 | 0.4471 | +0.0234 | +0.0129 |
| Swin Technique 2 | 0.4003 | 0.5903 | 0.4215 | +0.0178 | -0.0127 |
| Swin Technique 3 | 0.3731 | 0.5277 | 0.3850 | -0.0448 | -0.0492 |

Technique 1 is the only technique that improves both macro recall and macro F1 relative to Swin without techniques. Technique 2 improves macro recall but lowers macro F1, indicating that it likely increases sensitivity at the cost of precision. Technique 3 reduces both macro recall and macro F1, so it is not beneficial in its current configuration.

## 7. Performance Evaluation

The project evaluates models using macro precision, macro recall, macro F1, class-wise precision/recall/F1, confusion matrices, precision-recall curves, robustness comparison CSVs, and Grad-CAM visualizations.

Macro-averaged metrics are important because the dataset is highly imbalanced. Accuracy alone would be misleading because the dominant NV class is much larger than minority classes such as DF, VASC, and SCC. Macro recall and macro F1 give each class equal importance and better reflect performance on underrepresented diagnoses.

Precision measures the fraction of predicted positives for a class that are correct. Recall measures the fraction of true samples for a class that are recovered by the model. F1 is the harmonic mean of precision and recall. Macro precision, macro recall, and macro F1 are computed by calculating the metric independently for each class and then averaging across the eight classes with equal weight. Accuracy is the fraction of all test samples classified correctly. The confusion matrices are row-normalized by true class, and the precision-recall curves are generated in a one-vs-rest format for each class.

Grad-CAM is generated for the scratch-trained baseline and Swin models to support qualitative failure analysis. Foundation Grad-CAM is skipped because zero-shot CLIP does not use the same classifier and Grad-CAM path as the trained project models.

### Main Test Results

| Model / Run | Macro Precision | Macro Recall | Macro F1 |
| --- | ---: | ---: | ---: |
| Baseline ViT | 0.3849 | 0.4867 | 0.4048 |
| Swin no technique | 0.4048 | 0.5725 | 0.4342 |
| Swin Technique 1 | 0.4104 | 0.5960 | 0.4471 |
| Swin Technique 2 | 0.4003 | 0.5903 | 0.4215 |
| Swin Technique 3 | 0.3731 | 0.5277 | 0.3850 |
| Foundation CLIP zero-shot | 0.1683 | 0.1430 | 0.1297 |

### Swin Robustness Proxy Results

The robustness comparison shows identical full-test, no-ruler, and with-ruler metrics for each Swin run because the no-ruler and with-ruler splits are duplicated from the same test split.

| Run | Macro Recall | Macro F1 | Accuracy |
| --- | ---: | ---: | ---: |
| Swin no technique | 0.5725 | 0.4342 | 0.5821 |
| Swin Technique 1 | 0.5960 | 0.4471 | 0.5797 |
| Swin Technique 2 | 0.5903 | 0.4215 | 0.5817 |
| Swin Technique 3 | 0.5277 | 0.3850 | 0.4925 |

These results should be described as an evaluation consistency check rather than a true ruler/no-ruler robustness experiment.

### Graph and Visualization Analysis

Several graph-based artifacts were generated to support the quantitative evaluation: confusion matrices, precision-recall curves, epoch-trend plots, and Grad-CAM visualizations.

Recommended figures to include in the final formatted report:

| Figure | Suggested Content | Purpose |
| --- | --- | --- |
| Baseline confusion matrix | Baseline class-level confusion matrix | Shows baseline class-level errors. |
| Best model confusion matrix | Technique 1 class-level confusion matrix | Shows class-level behavior of the best trained model. |
| Foundation confusion matrix | Foundation-model confusion matrix | Shows zero-shot CLIP failure pattern. |
| Baseline PR curve | Baseline one-vs-rest PR curve | Shows baseline one-vs-rest class separation. |
| Best model PR curve | Technique 1 one-vs-rest PR curve | Shows PR behavior for the strongest trained run. |
| Foundation PR curve | Foundation-model PR curve | Shows weak zero-shot class separation. |
| Best model epoch trends | Technique 1 training and validation trends | Shows training/validation progress for the best run. |
| Example Grad-CAMs | Technique 1 Grad-CAM examples | Supports qualitative failure and attention analysis. |

Example embedded figures for drafting:

**Figure 1. Baseline confusion matrix**

![Baseline confusion matrix](outputs/baseline_none_best/confusion_matrix.png)

**Figure 2. Best-model confusion matrix (Technique 1)**

![Technique 1 confusion matrix](outputs/swin_technique1_best/confusion_matrix.png)

**Figure 3. Foundation-model confusion matrix**

![Foundation confusion matrix](outputs/foundation/confusion_matrix.png)

**Figure 4. Baseline precision-recall curves**

![Baseline precision-recall curve](outputs/baseline_none_best/pr_curve.png)

**Figure 5. Best-model precision-recall curves (Technique 1)**

![Technique 1 precision-recall curve](outputs/swin_technique1_best/pr_curve.png)

**Figure 6. Foundation-model precision-recall curves**

![Foundation precision-recall curve](outputs/foundation/pr_curve.png)

**Figure 7. Technique 1 training trend, epochs 1-10**

![Technique 1 epoch trend 1-10](outputs/swin_technique1_best/epoch_trend/swin_technique_1_1-10.png)

**Figure 8. Technique 1 training trend, epochs 11-50**

![Technique 1 epoch trend 11-50](outputs/swin_technique1_best/epoch_trend/swin_technique_1_11-50.png)

**Figure 9. Baseline Grad-CAM example: misclassified case, true BCC, predicted SCC**

![Baseline Grad-CAM example](outputs/baseline_none_best/gradcam/sample_0_true_2_pred_7.png)

**Figure 10. Technique 1 Grad-CAM example: correctly classified case, true BCC, predicted BCC**

![Technique 1 Grad-CAM correct example](outputs/swin_technique1_best/gradcam/correct_true_2_pred_2.png)

**Figure 11. Technique 1 Grad-CAM example: misclassified case, true BKL, predicted BCC**

![Technique 1 Grad-CAM example B](outputs/swin_technique1_best/gradcam/sample_22_true_4_pred_2.png)

The confusion matrices summarize class-level prediction behavior for each evaluated run. They are especially important for this project because the dataset is imbalanced and several minority classes have much lower sample counts than NV, MEL, and BCC. Across the trained models, the larger classes are generally more stable, while minority classes such as DF, SCC, AK, and VASC show more volatile behavior. This supports the decision to emphasize macro recall and macro F1 rather than relying only on accuracy. The confusion matrices should be used in the final formatted report to identify which classes are commonly confused and whether each technique shifts the model toward higher minority-class sensitivity or higher false-positive rates.

The precision-recall curves provide a complementary view of model performance under class imbalance. Because each class is evaluated in a one-vs-rest manner, these curves show how well each model separates each lesion category across different decision thresholds. The trained baseline and Swin models produce more meaningful precision-recall behavior than the zero-shot foundation model. The foundation model's weaker PR curves are consistent with its low macro F1, showing that zero-shot CLIP does not separate the ISIC diagnostic classes as effectively as task-trained models.

Epoch-trend plots were collected for the trained models to inspect training stability and convergence. These plots help verify that the models were actually trained from scratch and that validation performance was monitored over training rather than selecting checkpoints arbitrarily. The available trends support the final checkpointing strategy, where the best model is selected by validation macro recall. Technique 3 has a known logging gap for epochs 21-30 due to unrecoverable Colab logs, but its final checkpoints and evaluation artifacts were preserved.

Grad-CAM visualizations were generated for the baseline and Swin runs to support qualitative error analysis. These examples can be used to inspect whether a model's attention is concentrated on lesion regions, borders, or irrelevant image artifacts. For this project, Grad-CAM is especially relevant because the motivation is ruler-like border artifact bias. However, because the processed ISIC 2019 split does not include explicit ruler/no-ruler annotations, the Grad-CAM results should be interpreted qualitatively rather than as direct proof of ruler robustness. Foundation Grad-CAM is intentionally skipped because zero-shot CLIP does not use the same classifier path as the trained ViT/Swin models.

Overall, the graphs support the same conclusion as the metrics: Swin improves over the ViT baseline, Technique 1 provides the strongest overall trained result, Technique 2 improves recall but weakens F1 relative to Technique 1, Technique 3 is too aggressive in its current form, and zero-shot CLIP is not competitive for this specialized medical classification task.

## 8. Comparison of Approaches

The scratch-trained Swin Transformer without techniques improves over the scratch-trained ViT baseline. Baseline ViT achieved macro recall 0.4867 and macro F1 0.4048, while Swin without techniques achieved macro recall 0.5725 and macro F1 0.4342. This suggests that the hierarchical shifted-window design of Swin is better suited to this dataset than the baseline ViT configuration.

Technique 1 is the best-performing trained approach. It achieved macro recall 0.5960 and macro F1 0.4471. This indicates that border/ruler-focused preprocessing and synthetic augmentation improved the balance between detecting minority classes and maintaining useful precision.

Technique 2 achieved macro recall 0.5903, close to Technique 1, but macro F1 dropped to 0.4215. This suggests that attention regularization helped recall but may have introduced more false positives, lowering precision and F1.

Technique 3 underperformed all other Swin runs, with macro recall 0.5277, macro F1 0.3850, and accuracy 0.4925. The likely explanation is that masking border patch embeddings with probability 0.5 made the input representation too unstable or removed useful contextual features. Although the technique is motivated by bias mitigation, the implementation may require a lower mask probability, curriculum schedule, or more careful target-region selection.

The foundation CLIP model performed much worse than the trained models, with macro F1 0.1297. This is expected because CLIP was used as zero-shot inference without fine-tuning on ISIC 2019. Dermatology labels are subtle and domain-specific, and natural-language prompts may not align well with the visual distinctions required for skin lesion diagnosis.

Overall, the best model is Swin Technique 1 by macro recall and macro F1.

## 9. Foundation Model Comparison

The foundation model approach uses `openai/clip-vit-large-patch14` in zero-shot inference mode. CLIP is pretrained by design, but no project training or fine-tuning is performed. The model uses text prompts for each class and compares image embeddings against text embeddings.

The prompts used are:

- `a dermoscopic image of {}`
- `a photo of {}`
- `a skin lesion showing {}`

Class abbreviations are expanded into prompt-friendly names, such as MEL to melanoma, NV to melanocytic nevus, and BCC to basal cell carcinoma.

The foundation model's macro F1 is much lower than the trained models:

| Model | Macro Recall | Macro F1 |
| --- | ---: | ---: |
| Foundation CLIP zero-shot | 0.1430 | 0.1297 |
| Baseline ViT | 0.4867 | 0.4048 |
| Best Swin Technique 1 | 0.5960 | 0.4471 |

The most likely reasons for this gap are:

- CLIP is not trained specifically for ISIC 2019 diagnostic classification.
- Zero-shot prompts are too coarse for subtle dermatology categories.
- Some classes are visually similar and require domain-specific supervision.
- The dataset is highly imbalanced.
- CLIP's pretraining distribution differs from dermoscopic medical imaging.

Thus, CLIP is a useful foundation comparison but not competitive with task-trained transformer models in this setting.

## 10. Error and Failure Analysis

The class-wise metrics show that the models struggle most with minority classes and clinically subtle categories. DF, SCC, AK, and VASC have much smaller training counts than NV, MEL, and BCC. Even with class weighting, minority-class predictions remain unstable.

Baseline ViT performs reasonably on large classes such as NV but has weaker macro performance because of minority-class failures. Swin improves macro recall substantially, indicating better sensitivity across classes. However, Swin also shows precision-recall tradeoffs, especially for smaller classes. Some techniques increase recall by predicting minority classes more often, but this can lower precision.

Technique 1 provides the best tradeoff. It improves macro recall and macro F1, suggesting that the augmentation/preprocessing strategy supports generalization without excessively destabilizing predictions.

Technique 2 improves recall but lowers F1 relative to Technique 1, which suggests a higher false-positive burden. Attention regularization may have successfully reduced border attention, but it may also have weakened useful attention patterns.

Technique 3 is the clearest failure case. Its lower accuracy and macro F1 suggest that aggressive border masking removed information that the model still needed or made training less stable.

Concrete quantitative failure examples:

| Model / Run | Failure Pattern | Evidence |
| --- | --- | --- |
| Baseline ViT | Weak minority-class F1 despite reasonable NV performance | DF F1 = 0.1500 and SCC F1 = 0.1693, while NV F1 = 0.7685 |
| Swin no technique | Better recall than baseline but still unstable minority-class precision | DF precision = 0.1628 and SCC precision = 0.1883 |
| Swin Technique 1 | Best overall tradeoff, but minority precision remains limited | Macro F1 improves to 0.4471, but AK precision = 0.2637 and SCC precision = 0.2129 |
| Swin Technique 2 | Recall-oriented behavior with lower F1 than Technique 1 | VASC recall = 1.0000, but VASC precision = 0.2427 and macro F1 = 0.4215 |
| Swin Technique 3 | Border masking appears too aggressive | Accuracy drops to 0.4925 and macro F1 drops to 0.3850 |
| Foundation CLIP | Zero-shot model fails on several medical classes | VASC F1 = 0.0000 and SCC F1 = 0.0000 |

These examples show that the strongest model is not simply the one with the highest recall on a single minority class. The best model should balance recall and precision across all eight classes. By that criterion, Swin Technique 1 is the strongest trained run because it has the best macro F1 and macro recall together.

Grad-CAM visualizations are available for the baseline and Swin runs. These examples can be used to show whether the trained models focus on lesion regions, borders, or unrelated artifacts. Because foundation is zero-shot CLIP and does not use the same classifier path, Grad-CAM is intentionally skipped for the foundation model.

Important limitation: the project does not have explicit ruler/no-ruler annotations. Therefore, the no-ruler and with-ruler robustness comparisons cannot be used as real evidence that a model is robust to rulers. They should be presented as an evaluation consistency check, while the main bias-mitigation evidence should come from the technique design, quantitative model comparisons, confusion matrices, PR curves, and Grad-CAM qualitative analysis.

## 11. Team Collaboration

[Placeholder: Add how the team coordinated the project. For example: The group shared the dataset split, baseline model, and foundation model setup. Team members coordinated through shared repo artifacts and consistent train/validation/test CSVs so each model used the same evaluation protocol.]

[Placeholder: Add who handled dataset preparation, baseline training, Swin training, technique implementation, foundation evaluation, result analysis, README/checkpoint packaging, and report/presentation preparation.]

## 12. Individual Contributions

[Placeholder: Replace with actual member names and contributions.]

| Team Member | Contributions |
| --- | --- |
| Member 1 | Dataset setup, split generation, baseline/foundation/shared infrastructure, or other assigned work. |
| Member 2 | Swin model training, technique implementation, evaluation, or other assigned work. |
| Member 3 | Additional technique, analysis, visualization, report, or presentation work. |

Each member should be credited with at least one trained or improved model component, in line with the project requirement that each member train at least one model or contribute a distinct proposed-model improvement.

## 13. Conclusion

This project implemented and evaluated transformer-based skin lesion classifiers on ISIC 2019. The baseline ViT and proposed Swin Transformer were both trained from scratch, satisfying the no-pretrained-model requirement for project-trained models. The foundation model was used only as zero-shot CLIP inference and was not trained or fine-tuned.

The proposed Swin Transformer outperformed the baseline ViT. Among the three improvement techniques, Technique 1 achieved the best result with macro recall 0.5960 and macro F1 0.4471. Technique 2 improved recall but did not match Technique 1 in F1, while Technique 3 underperformed and likely masked too much useful image context. The foundation CLIP model performed substantially worse than the trained models, which is expected for zero-shot inference on a specialized medical-imaging task.

The main limitation is that the processed ISIC 2019 setup does not include explicit ruler/no-ruler annotations. Therefore, this project should be framed as ruler-like border artifact mitigation rather than a definitive ruler-bias measurement study. Future work should use a curated ruler-annotated test set, tune the strength of border masking and attention regularization, and explore more clinically grounded prompts or fine-tuned medical foundation models.

Despite this limitation, the project demonstrates a complete transformer classification pipeline with scratch-trained models, consistent splits, multiple improvement techniques, foundation-model comparison, quantitative metrics, visual artifacts, and a clear analysis path for a graduate-level deep learning report.
