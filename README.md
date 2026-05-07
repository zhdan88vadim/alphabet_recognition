
<!-- 

tensorboard --logdir=runs

conda activate /mnt/ntfs/learn_ML/test_classes/Тестовое\ Python\ ML,CV/Тестовое_ML/тестовое_ml/.conda 

-->


## 📊 Training Results

### Loss Curves
![Training and Validation Loss](readme_images/tb_loss.png)

*Training completed in 20 epochs with early stopping*

### Accuracy
![Accuracy Plot](readme_images/tb_accuracy.png)

- **Best Validation Accuracy**: 98.42%
- **Test Accuracy**: ---%

### Data Augmentation Examples
![Augmented Samples](readme_images/tb_aug_images.png)

## 🔍 Recognition Examples

| Original Image | Recognized Result |
|:--------------:|:-----------------:|
| ![Original 1](readme_images/4_1.png) | ![Original 1](readme_images/4.png) |
| ![Original 2](readme_images/1_1.png) | ![Original 2](readme_images/1.png) |
| ![Original 3](readme_images/2_1.png) | ![Original 3](readme_images/2.png) |
| ![Original 4](readme_images/3_1.png) | ![Original 4](readme_images/3.png) |

## Confusion matrix
![Confusion matrix](readme_images/confusion_matrix.png)

## 🏃‍♂️ Reproduce Results

```bash
conda env create -f environment.yaml
conda activate alphabet_env
python train.py
```


## 📋 TODO (Priority: 🔴 High → 🟡 Medium → 🟢 Low)

### Data Analysis
- 🔴 [ ] Add EDA results (class distribution, image statistics, sample visualizations)
- 🔴 [ ] Analyze misclassifications (which letters are most confused)
- 🟡 [ ] Add class weights to handle imbalanced data

### Baseline Models
- 🔴 [ ] Compare with HOG + SVM or RandomForest

### Model & Training Improvements
- 🔴 [ ] Check accuracy with smaller image size (32×32)
- 🟡 [ ] Add learning rate scheduling
- 🟡 [ ] Experiment with different optimizers (AdamW, SGD with momentum)
- 🟢 [ ] Implement k-fold cross validation
- 🟢 [ ] Add label smoothing
- 🟢 [ ] Implement mixup or cutmix augmentation
- 🟢 [ ] Add gradient clipping

### Evaluation & Metrics
- 🔴 [ ] Plot confusion matrix
- 🔴 [ ] Add precision, recall, F1-score per class
- 🟡 [ ] Calculate inference time (FPS) on CPU and GPU
- 🟢 [ ] Add top-2 and top-3 accuracy
- 🟢 [ ] Add ROC curves and AUC for each class

### Visualization & Debugging
- 🟡 [ ] Add Grad-CAM visualization
- 🟡 [ ] Save misclassified examples with predictions
- 🟢 [ ] Add model architecture diagram
- 🟢 [ ] Plot feature embeddings with t-SNE or UMAP
- 🟢 [ ] Add learning rate vs loss plot

### Environment & Reproducibility
- 🔴 [ ] Test environment files on a clean machine