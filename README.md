
<!-- 

tensorboard --logdir=runs

conda activate /mnt/ntfs/learn_ML/test_classes/Тестовое\ Python\ ML,CV/Тестовое_ML/тестовое_ml/.conda 

-->

## General Description

I took on a test assignment to recognize letters in images. A dataset was provided for this task.

After analyzing the dataset, I noticed that the letters in it have **thick lines**, but the actual task requires recognizing **thin letters written with a regular pen**.

<img src="readme_images/original_test_img.png" width="50%">

### The Problem

| Aspect | Dataset | Target |
|--------|---------|--------|
| Line thickness | Thick | Thin (pen-written) |
| Style | Bold/display font | Handwriting-like |

This is a **domain shift** problem — the training data distribution differs from the real inference data.

### My Solution


To address this, I added a **preprocessing step** that converts thick letters to thin ones. This transformation was applied to **both training and validation data**.

Additionally, the dataset had large empty margins around the letters. I cropped out the excess whitespace to maximize useful data while significantly saving memory.

### Cropping vs Padding

| Technique | Description | When to use |
|-----------|-------------|-------------|
| **Cropping** | Remove empty borders around the letter | When whitespace is excessive and offers no value |
| **Padding** | Add borders around the image | To preserve aspect ratio or create square images |

**My approach:** I used cropping to eliminate unnecessary empty space, then applied padding to create uniform square images before resizing.

### Complete preprocessing pipeline:

1. **Crop** — remove empty margins around the letter
2. **Padding** — add minimal borders to preserve letter proportions (avoid distortion)
3. **Thinning** — convert thick strokes to thin (pen-like)
4. **Resize** — to 64×64 for training

### Benefits of my approach:

- Preserves letter structure
- Reduces noise (empty space)
- Maintains aspect ratio
- Saves memory
- Forces model to focus only on relevant features


### Results

This approach showed **quite good results**:
- The model learned to recognize features of thin letters
- Validation performance improved significantly
- Better generalization to real pen-written letters


## EDA Report for Letter Dataset

### 1. Image Dimensions
- **All images have the same size:** 278×278 pixels
- **To save memory and speed up training** we will resize images to 64×64 pixels (preserving aspect ratio)

### 2. Class Imbalance
- **Total images:** 33,141
- **Mean per class:** 1,004.3
- **Minimum class:** Б (496 samples)
- **Maximum class:** Я (1,200 samples)
- **Imbalance ratio:** 2.42

**Impact assessment:** For a convolutional network, an imbalance of 2.42 is considered **moderate** and not critical. Modern CNNs are quite robust to this ratio.

**Solution:** Leave as is for now. If the model performs poorly on class Б, we will add:
- Class weights
- Augmentation for minority classes

### 3. Training Strategy
- **Not much data** (33k images) — risk of overfitting
- Will use a **simple convolutional network** (3-4 conv layers)
- Required regularization methods:
  - Dropout (0.3-0.5)
  - Early stopping
  - Data augmentation

### 4. Final Plan
1. Resize images to 64×64
2. Normalize pixels to [0, 1]
3. Train/val/test split (70/15/15)
4. Train simple CNN with regularization
5. Monitor performance on class Б

### 5. Risks
- **Overfitting** — due to limited amount of data
- **Class Б** — least represented (496 samples)
- **Mitigation:** augmentation + dropout + early stopping

## Class distribution
![Class distribution](readme_images/eda_class_distribution.png)

## Examples of letters from the dataset
![Examples of letters from the dataset](readme_images/eda_visualize_samples_per_class.png)


### Data Augmentation Examples
<img src="readme_images/tb_aug_images.png" width="50%">
<!-- ![Augmented Samples](readme_images/tb_aug_images.png) -->



## 📊 Training Results

### Loss Curves
![Training and Validation Loss](readme_images/training_validation_loss.png)

*Training completed with early stopping*

### Accuracy
![Accuracy Plot](readme_images/training_validation_accuracy.png)

- **Best Validation Accuracy**: 90.56%
- **Best Validation F1 Score**: 0.89


### Train Confusion matrix
![Confusion matrix](readme_images/confusion_matrix_train.png)


## 🔍 Recognition Examples


![0](readme_images/predicted/0__predicted.png)
![1](readme_images/predicted/1__predicted.png)
![2](readme_images/predicted/2__predicted.png)
<!-- ![3](readme_images/predicted/3__predicted.png) -->
![4](readme_images/predicted/4__predicted.png)
![11](readme_images/predicted/11__predicted.png)
![12](readme_images/predicted/12__predicted.png)
![13](readme_images/predicted/13__predicted.png)
![14](readme_images/predicted/14__predicted.png)
![15](readme_images/predicted/15__predicted.png)
![n0](readme_images/predicted/n0__predicted.png)
<!-- ![n1](readme_images/predicted/n1__predicted.png) -->
![n2](readme_images/predicted/n2__predicted.png)
![n3](readme_images/predicted/n3__predicted.png)
![t0](readme_images/predicted/t0__predicted.png)
![t1](readme_images/predicted/t1__predicted.png)
![t2](readme_images/predicted/t2__predicted.png)
![t3](readme_images/predicted/t3__predicted.png)
![t4](readme_images/predicted/t4__predicted.png)
![t5](readme_images/predicted/t5__predicted.png)
![t6](readme_images/predicted/t6__predicted.png)
![t7](readme_images/predicted/t7__predicted.png)
![test_letters_my0](readme_images/predicted/test_letters_my0__predicted.png)
![test_my](readme_images/predicted/test_my__predicted.png)


## Model Robustness Testing

Testing the CNN model's resistance to image distortions.

### Transformations Applied

| Distortion | Range | Description |
|------------|-------|-------------|
| **Translation** | ±5-10 px | Shifting from center |
| **Rotation** | -20° to +20° | Angular distortion |
| **Scaling** | 0.5× to 1.2× | Size variation |


![Example predictions](readme_images/validation_cnn_distorted_predictions.png)
*Figure 1: Example predictions on distorted validation images showing model performance on augmented data*

![Confusion matrix](readme_images/validation_cnn_confusion_matrix.png)
*Figure 2: Confusion matrix showing classification errors between letter classes*

![Misclassified examples](readme_images/validation_cnn_misclassified.png)
*Figure 3: Examples of misclassified letters with predicted vs actual labels*


## Validation Classification Report

| Letter | precision | recall | f1-score | support |
|--------|-----------|--------|----------|---------|
| Ё | 0.86 | 0.86 | 0.86 | 21 |
| А | 0.87 | 0.81 | 0.84 | 32 |
| Б | 0.84 | 1.00 | 0.91 | 16 |
| В | 1.00 | 1.00 | 1.00 | 15 |
| Г | 0.87 | 0.87 | 0.87 | 15 |
| Д | 0.92 | 1.00 | 0.96 | 12 |
| Е | 0.85 | 0.85 | 0.85 | 20 |
| Ж | 0.71 | 1.00 | 0.83 | 10 |
| З | 1.00 | 1.00 | 1.00 | 15 |
| И | 0.72 | 0.59 | 0.65 | 22 |
| Й | 0.92 | 0.60 | 0.73 | 20 |
| К | 1.00 | 1.00 | 1.00 | 15 |
| Л | 0.96 | 0.75 | 0.84 | 32 |
| М | 0.79 | 0.96 | 0.86 | 23 |
| Н | 1.00 | 0.62 | 0.77 | 16 |
| О | 0.84 | 1.00 | 0.91 | 16 |
| П | 0.58 | 1.00 | 0.74 | 7 |
| Р | 1.00 | 1.00 | 1.00 | 7 |
| С | 1.00 | 1.00 | 1.00 | 15 |
| Т | 1.00 | 0.93 | 0.96 | 14 |
| У | 0.88 | 1.00 | 0.93 | 7 |
| Ф | 1.00 | 1.00 | 1.00 | 8 |
| Х | 0.89 | 1.00 | 0.94 | 8 |
| Ц | 1.00 | 1.00 | 1.00 | 7 |
| Ч | 0.25 | 0.50 | 0.33 | 2 |
| Ш | 0.88 | 1.00 | 0.93 | 7 |
| Щ | 1.00 | 1.00 | 1.00 | 6 |
| Ъ | 1.00 | 1.00 | 1.00 | 11 |
| Ы | 0.92 | 0.85 | 0.88 | 13 |
| Ь | 0.93 | 0.87 | 0.90 | 31 |
| Э | 1.00 | 1.00 | 1.00 | 18 |
| Ю | 0.50 | 1.00 | 0.67 | 4 |
| Я | 1.00 | 1.00 | 1.00 | 12 |
| **accuracy** | | | **0.89** | **477** |
| **macro avg** | **0.88** | **0.91** | **0.88** | **477** |
| **weighted avg** | **0.90** | **0.89** | **0.89** | **477** |


<br>  

![Validation metrics](readme_images/metrics/val/per_letter_metrics.png)


## Reproduce Results

```bash
conda env create -f environment.yaml
conda activate alphabet_env

mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

python train.py
```


## 📋 TODO (Priority: 🔴 High → 🟡 Medium → 🟢 Low)

### Data Analysis
- [x] Add EDA results (class distribution, image statistics, sample visualizations)
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
- [x] Plot confusion matrix
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