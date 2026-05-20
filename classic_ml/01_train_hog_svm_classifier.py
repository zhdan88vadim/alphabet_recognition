import os
from augmentation import AdaptiveAugmentationBuilder, CenterDigitsTransform, ExtractLetterWithMargin, Invert, SquarePad
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torchvision import datasets
from torchvision import transforms
from PIL import Image
import json
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import torch
import warnings
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
warnings.filterwarnings('ignore')

DATA_ROOT = "../dataset"
# DATA_ROOT = "/mnt/ntfs/learn_ML/test_classes/Тестовое Python ML,CV/Тестовое_ML/тестовое_ml/dataset/test_unique_only/"



def visualize_class_distances(X, y, class_names, pipeline):
    """
    Calculate and visualize distances between class centers.
    Shows which classes are close (often confused).
    """

    from sklearn.metrics.pairwise import euclidean_distances
    
    unique_classes = np.unique(y)
    class_centers = {}
    
    for class_id in unique_classes:
        mask = y == class_id
        class_centers[class_id] = X[mask].mean(axis=0)
    
    center_matrix = np.array([class_centers[c] for c in unique_classes])
    distances = euclidean_distances(center_matrix, center_matrix)


    # Create distance matrix visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Distance heatmap
    class_labels = [class_names[c] for c in unique_classes]
    im = axes[0].imshow(distances, cmap='hot', interpolation='nearest')
    axes[0].set_xticks(range(len(class_labels)))
    axes[0].set_yticks(range(len(class_labels)))
    axes[0].set_xticklabels(class_labels, rotation=90, fontsize=8)
    axes[0].set_yticklabels(class_labels, fontsize=8)
    axes[0].set_title('Euclidean Distance Between Class Centers')
    plt.colorbar(im, ax=axes[0])
    
    # Add distance values in cells
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            if i != j:
                text = axes[0].text(j, i, f'{distances[i, j]:.1f}',
                                   ha="center", va="center", color="white", fontsize=6)
    
    # Plot 2: Find and display closest class pairs
    closest_pairs = []
    for i in range(len(unique_classes)):
        for j in range(i+1, len(unique_classes)):
            closest_pairs.append((class_labels[i], class_labels[j], distances[i, j]))
    
    closest_pairs.sort(key=lambda x: x[2])
    
    # Create table of closest classes
    table_data = []
    for pair in closest_pairs[:15]:  # Top 15 closest pairs
        table_data.append([pair[0], pair[1], f'{pair[2]:.2f}'])
    
    table = axes[1].table(cellText=table_data,
                          colLabels=['Class A', 'Class B', 'Distance'],
                          cellLoc='center',
                          loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    axes[1].axis('off')
    axes[1].set_title('Closest Class Pairs (Most Likely to be Confused)')
    
    plt.tight_layout()
    plt.savefig('class_distances.png', dpi=150)
    plt.show()
    
    print(f"\n📊 Top 10 closest class pairs (most similar):")
    for i in range(min(10, len(closest_pairs))):
        class1, class2, dist = closest_pairs[i]
        print(f"   {class1} ↔ {class2}: {dist:.2f}")
    
    return distances, closest_pairs


# ============================================
# 1. HOG DESCRIPTOR SETUP
# ============================================

def create_hog_descriptor(image_size=(64, 64)):
    """
    Create HOG descriptor optimized for letter recognition.
    """
    win_size = image_size
    cell_size = (8, 8)
    block_size = (16, 16)
    block_stride = (8, 8)
    nbins = 9
    
    hog = cv2.HOGDescriptor(
        _winSize=win_size,
        _blockSize=block_size,
        _blockStride=block_stride,
        _cellSize=cell_size,
        _nbins=nbins
    )
    return hog

def extract_hog_features(images, hog=None, image_size=(64, 64), verbose=True):
    """
    Extract HOG features from images.
    
    Args:
        images: List of numpy arrays
        hog: HOG descriptor (creates new if None)
        image_size: Target size (width, height)
        verbose: Print progress
    
    Returns:
        features: numpy array of shape (n_samples, feature_dim)
    """
    if hog is None:
        hog = create_hog_descriptor(image_size)
    
    n_samples = len(images)
    feature_dim = hog.getDescriptorSize()
    
    if verbose:
        print(f"   Feature dimension: {feature_dim}")
        print(f"   Processing {n_samples} images...")
    
    features = np.zeros((n_samples, feature_dim), dtype=np.float32)
    
    for i in range(n_samples):
        img = images[i]
        
        # Convert to uint8 if needed
        if img.dtype != np.uint8:
            if img.max() <= 1:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        
        # Ensure 2D grayscale
        if len(img.shape) == 3:
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif img.shape[2] == 1:
                img = img.squeeze()
        
        # Resize if needed
        if img.shape != image_size:
            img = cv2.resize(img, image_size)

        # Ensure contiguous array (fixes some OpenCV issues)
        img = np.ascontiguousarray(img)
        
        # Compute HOG features
        try:
            feat = hog.compute(img)
            features[i] = feat.flatten()
        except Exception as e:
            if verbose and i < 5:
                print(f"   Warning: Error on image {i}: {e}")
            features[i] = np.zeros(feature_dim)
        
        # Progress indicator
        if verbose and (i + 1) % 5000 == 0:
            print(f"   Processed {i + 1}/{n_samples} images...")
    
    if verbose:
        print(f"   Done! Features shape: {features.shape}")
    
    return features

import yaml

def load_config(config_path='config.yaml'):
    """Loads the configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# ============================================
# 2. DATA LOADING WITH PROPER TRAIN/VAL/TEST SPLIT
# ============================================
def create_train_val_test_datasets(data_root, test_size=0.15, val_size=0.15):
    """
    Creates train, validation, and test datasets with proper transforms.
    """
    config = load_config()
    aug_builder = AdaptiveAugmentationBuilder(base_size=config['data']['image_size'])
    
    # Transform for training (with augmentation)
    train_transform = aug_builder.build_train_transform(
        (config['data']['image_size'], config['data']['image_size'])
    )
    
    # Transform for validation and test (without augmentation)
    val_test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        ExtractLetterWithMargin(margin=4, fill_white=True),
        # CenterDigitsTransform(padding=2, fill_value=255),
        SquarePad(fill_white=True),
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        Invert(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Загружаем ДВА РАЗНЫХ датасета
    full_dataset_train = datasets.ImageFolder(root=data_root, transform=train_transform)
    full_dataset_val_test = datasets.ImageFolder(root=data_root, transform=val_test_transform)
    
    class_names = full_dataset_train.classes
    
    # Получаем индексы для разделения (на основе train датасета)
    n_total = len(full_dataset_train)
    indices = list(range(n_total))
    
    # Стратифицированное разделение
    from collections import defaultdict
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(full_dataset_train):
        class_indices[label].append(idx)
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    for label, idx_list in class_indices.items():
        n_class = len(idx_list)
        n_test = int(n_class * test_size)
        n_val = int(n_class * val_size)
        n_train = n_class - n_test - n_val
        
        np.random.seed(42)
        shuffled = np.random.permutation(idx_list)
        
        train_indices.extend(shuffled[:n_train])
        val_indices.extend(shuffled[n_train:n_train+n_val])
        test_indices.extend(shuffled[n_train+n_val:])
    
    # Создаем подмножества из разных датасетов
    train_dataset = torch.utils.data.Subset(full_dataset_train, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset_val_test, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset_val_test, test_indices)
    
    # Конвертируем в numpy
    train_images, train_labels = dataset_to_numpy(train_dataset)
    val_images, val_labels = dataset_to_numpy(val_dataset)
    test_images, test_labels = dataset_to_numpy(test_dataset)
    
    return train_images, train_labels, val_images, val_labels, test_images, test_labels, class_names

def dataset_to_numpy(dataset):
    """Convert torch Dataset to numpy arrays."""
    images = []
    labels = []
    
    for i in range(len(dataset)):
        img_tensor, label = dataset[i]
        # Convert tensor to numpy (1, 64, 64) -> (64, 64)
        img_np = img_tensor.squeeze().cpu().numpy()
        # Scale from [0, 1] to [0, 255] for HOG
        img_np = (img_np * 255).astype(np.uint8)

        # INVERT FOR HOG
        # img_np = cv2.bitwise_not(img_np)
        images.append(img_np)
        labels.append(label)
    
    return np.stack(images), np.array(labels)

# ============================================
# 3. TRAIN HOG + SVM CLASSIFIER
# ============================================

def train_hog_svm(X_train, y_train, use_pca=True, pca_components=100):
    """
    Train SVM classifier on HOG features.
    
    Args:
        X_train: Training HOG features
        y_train: Training labels
        use_pca: Whether to apply PCA for dimensionality reduction
        pca_components: Number of PCA components
    
    Returns:
        clf: Trained SVM classifier
        scaler: StandardScaler
        pca: PCA transformer (if used)
    """
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=min(pca_components, X_train.shape[1]))),
        ('svm', svm.SVC(C=10, gamma=0.001, kernel='rbf', probability=True, random_state=42, class_weight='balanced'))
    ])
    
    cv_score_correct = cross_val_score(pipeline, X_train, y_train, cv=3, n_jobs=-1).mean()
    print(f"   Baseline CV accuracy: {cv_score_correct:.4f} ({cv_score_correct*100:.2f}%)")

    param_grid = {
        'svm__C': [10],
        'svm__gamma': [0.001],
        'svm__kernel': ['rbf']
    }
    
    # param_grid = {
    #     'svm__C': [1, 10, 100],
    #     'svm__gamma': [0.1, 0.01, 0.001],
    #     'svm__kernel': ['rbf']
    # }
    
    grid_search = GridSearchCV(
        pipeline, 
        param_grid,
        cv=3, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    print("   Performing grid search...")
    grid_search.fit(X_train, y_train)
    
    print(f"   Best parameters: {grid_search.best_params_}")
    print(f"   Best CV accuracy: {grid_search.best_score_:.4f} ({grid_search.best_score_*100:.2f}%)")
   
    # Print explained variance if PCA was used
    best_pipeline = grid_search.best_estimator_
    if use_pca and 'pca' in best_pipeline.named_steps:
        pca_step = best_pipeline.named_steps['pca']
        explained_var = pca_step.explained_variance_ratio_.sum()
        print(f"   PCA explained variance: {explained_var:.2%}")
    
    return best_pipeline

# ============================================
# 4. VISUALIZATION FUNCTIONS
# ============================================

def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix_hog_svm.png', title='Confusion Matrix'):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{title} - HOG + SVM')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"   Confusion matrix saved to {save_path}")
    return cm

def plot_top_misclassifications(y_true, y_pred, class_names, top_k=10, title="Misclassifications"):
    """Print most frequently misclassified pairs."""
    from collections import Counter
    
    misclassifications = []
    for true, pred in zip(y_true, y_pred):
        if true != pred:
            misclassifications.append((class_names[true], class_names[pred]))
    
    counter = Counter(misclassifications)
    
    print(f"\n📊 {title} - Top {top_k} misclassifications:")
    for (true, pred), count in counter.most_common(top_k):
        print(f"   {true} → {pred}: {count} times")
    
    return counter

def visualize_sample_images(images, labels, class_names, num_samples=50, title="Sample Images", selected_classes=None):
    """
    Visualize sample images from dataset with option to filter by classes.
    
    Args:
        images: numpy array of images
        labels: numpy array of labels
        class_names: list of class names
        num_samples: number of samples to display
        title: plot title
        selected_classes: list of class names to show (e.g., ['1', '7'] or None for all classes)
    """
    # Фильтрация по выбранным классам
    if selected_classes is not None:
        # Находим индексы выбранных классов
        selected_indices = []
        for class_name in selected_classes:
            if class_name in class_names:
                class_idx = class_names.index(class_name)
                selected_indices.extend(np.where(labels == class_idx)[0])
        
        if len(selected_indices) == 0:
            print(f"⚠️ Ни одного изображения не найдено для классов: {selected_classes}")
            return
        
        # Фильтруем изображения и метки
        filtered_images = images[selected_indices]
        filtered_labels = labels[selected_indices]
        
        # Обновляем количество samples
        num_samples = min(num_samples, len(filtered_images))
        indices = np.random.choice(len(filtered_images), num_samples, replace=False)
        
        display_images = filtered_images[indices]
        display_labels = filtered_labels[indices]
        
        print(f"📊 Показано {num_samples} изображений из классов: {selected_classes}")
    else:
        # Показываем все классы
        num_samples = min(num_samples, len(images))
        indices = np.random.choice(len(images), num_samples, replace=False)
        display_images = images[indices]
        display_labels = labels[indices]
    
    # Расчет сетки
    n_cols = 10
    n_rows = (num_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 2*n_rows))
    axes = axes.flatten()
    
    # Отображаем изображения
    for i in range(num_samples):
        img = display_images[i]
        label = class_names[display_labels[i]]
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'{label}', fontsize=8)
        axes[i].axis('off')
    
    # Скрываем лишние подграфики
    for j in range(num_samples, len(axes)):
        axes[j].axis('off')
    
    # Обновляем заголовок
    if selected_classes:
        title = f"{title} - Classes: {', '.join(selected_classes)}"
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return display_images, display_labels

# def visualize_sample_images(images, labels, class_names, num_samples=50, title="Sample Images"):
#     """Visualize sample images from dataset."""
#     indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
    
#     n_cols = 10
#     n_rows = (num_samples + n_cols - 1) // n_cols
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 2*n_rows))
#     axes = axes.flatten()
    
#     for i, idx in enumerate(indices):
#         img = images[idx]
#         label = class_names[labels[idx]]
        
#         axes[i].imshow(img, cmap='gray')
#         axes[i].set_title(f'{label}', fontsize=8)
#         axes[i].axis('off')
    
#     for j in range(len(indices), len(axes)):
#         axes[j].axis('off')
    
#     plt.suptitle(title, fontsize=16)
#     plt.tight_layout()
#     plt.show()

def visualize_hog_features(image, hog, save_path='hog_visualization.png'):
    """Visualize HOG features for a single image."""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    image = cv2.resize(image, (64, 64))
    
    # Compute HOG features
    hog_features = hog.compute(image).flatten()
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # HOG gradient magnitude visualization
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    
    axes[1].imshow(mag, cmap='hot')
    axes[1].set_title('HOG Gradient Magnitude')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"   HOG visualization saved to {save_path}")

# ============================================
# 5. SAVE AND LOAD MODEL
# ============================================

def save_model(pipeline, class_names, train_accuracy, val_accuracy, test_accuracy, filepath):
    """
    Save model artifacts (excluding unpicklable HOG object).
    """
    print(f"\n💾 Saving model to {filepath}...")
    
    model_artifacts = {
        'pipeline': pipeline, 
        'class_names': class_names,
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy,
        'hog_params': {
            'win_size': (64, 64),
            'block_size': (16, 16),
            'block_stride': (8, 8),
            'cell_size': (8, 8),
            'nbins': 9
        }
    }
    
    try:
        joblib.dump(model_artifacts, filepath)
        file_size = os.path.getsize(filepath) / 1024 / 1024
        print(f"   ✓ Model saved successfully to {filepath}")
        print(f"   File size: {file_size:.2f} MB")
    except Exception as e:
        print(f"   ✗ Error saving: {e}")
        # Alternative: save without clf if needed
        print("   Trying alternative save method...")
        joblib.dump({
            'pipeline': pipeline,
            'class_names': class_names,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,
            'hog_params': model_artifacts['hog_params']
        }, filepath.replace('.pkl', '_simple.pkl'))

def load_model(filepath='hog_svm_alphabet_model.pkl'):
    """
    Load model and recreate HOG descriptor.
    """
    print(f"📂 Loading model from {filepath}...")
    
    if not os.path.exists(filepath):
        print(f"   ✗ Model file {filepath} not found!")
        return None
    
    try:
        artifacts = joblib.load(filepath)
        
        # Recreate HOG descriptor from saved parameters
        if 'hog_params' in artifacts:
            params = artifacts['hog_params']
            hog = cv2.HOGDescriptor(
                _winSize=params['win_size'],
                _blockSize=params['block_size'],
                _blockStride=params['block_stride'],
                _cellSize=params['cell_size'],
                _nbins=params['nbins']
            )
            artifacts['hog'] = hog
        
        print(f"   ✓ Model loaded successfully")
        if 'test_accuracy' in artifacts:
            print(f"   Test accuracy: {artifacts['test_accuracy']*100:.2f}%")
        return artifacts
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        return None

# ============================================
# 6. MAIN PIPELINE WITH PROPER TRAIN/VAL/TEST
# ============================================

def main():
    train_images, train_labels, val_images, val_labels, test_images, test_labels, class_names = create_train_val_test_datasets(
        DATA_ROOT, test_size=0.15, val_size=0.15
    )

    print("\n📸 Visualizing sample images...")
    visualize_sample_images(train_images, train_labels, class_names, 
                           num_samples=30, title="Training Set Samples")
    visualize_sample_images(val_images, val_labels, class_names, 
                           num_samples=20, title="Validation Set Samples")
    visualize_sample_images(test_images, test_labels, class_names, 
                           num_samples=20, title="Test Set Samples")
    
    # ============================================
    # Extract HOG features
    # ============================================
    print("\n🔧 Extracting HOG features...")
    hog = create_hog_descriptor(image_size=(64, 64))
    
    print("\n   Training set:")
    X_train = extract_hog_features(train_images, hog, verbose=True)
    print("\n   Validation set:")
    X_val = extract_hog_features(val_images, hog, verbose=True)
    print("\n   Test set:")
    X_test = extract_hog_features(test_images, hog, verbose=True)
    
    print(f"\n   Feature vector size: {X_train.shape[1]}")
    
    # ============================================
    # Train HOG + SVM on TRAINING set only
    # ============================================
    print("\n🏋️ Training HOG + SVM on training set...")
    pipeline = train_hog_svm(
        X_train, train_labels, 
        use_pca=True, 
        pca_components=100
    )
    
    print("\n📈 Evaluating on training set...")
    # y_train_pred = pipeline.predict(X_train)

    train_proba = pipeline.predict_proba(X_train)
    y_train_pred = np.argmax(train_proba, axis=1)

    train_accuracy = accuracy_score(train_labels, y_train_pred)
    print(f"   Training Accuracy: {train_accuracy*100:.2f}%")
    
    print("\n📈 Evaluating on validation set...")
    # y_val_pred = pipeline.predict(X_val)

    val_proba = pipeline.predict_proba(X_val)
    y_val_pred = np.argmax(val_proba, axis=1)

    val_accuracy = accuracy_score(val_labels, y_val_pred)
    print(f"   Validation Accuracy: {val_accuracy*100:.2f}%")


    print("\n🎯 FINAL EVALUATION ON TEST SET:")
    print("   (This should be done only once after model is finalized)")
    y_test_pred = pipeline.predict(X_test)

    test_proba = pipeline.predict_proba(X_test)
    y_test_pred = np.argmax(test_proba, axis=1)

    test_accuracy = accuracy_score(test_labels, y_test_pred)
    
    print(f"\n{'='*60}")
    print(f"✅ FINAL TEST ACCURACY: {test_accuracy*100:.2f}%")
    print(f"{'='*60}")
    
    # ============================================
    # Detailed classification reports
    # ============================================
    print("\n📋 Classification Report - Validation Set:")
    val_report = classification_report(val_labels, y_val_pred, target_names=class_names, output_dict=True)
    print(f"   Macro avg - Precision: {val_report['macro avg']['precision']:.3f}, "
          f"Recall: {val_report['macro avg']['recall']:.3f}, "
          f"F1: {val_report['macro avg']['f1-score']:.3f}")
    
    print("\n📋 Classification Report - Test Set:")
    test_report = classification_report(test_labels, y_test_pred, target_names=class_names, output_dict=True)
    print(f"   Macro avg - Precision: {test_report['macro avg']['precision']:.3f}, "
          f"Recall: {test_report['macro avg']['recall']:.3f}, "
          f"F1: {test_report['macro avg']['f1-score']:.3f}")
    


    misclassified_idx = np.where((test_labels != y_test_pred))[0]

    if len(misclassified_idx) > 0:
        print(f"\n🔍 Анализ первых {min(4, len(misclassified_idx))} ошибочных предсказаний:")
        
        # Показываем первые 50 ошибок
        for idx_num, idx in enumerate(misclassified_idx[:4]):
            print(f"\n--- Ошибка {idx_num + 1} (Индекс: {idx}) ---")
            
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(test_images[idx], cmap='gray')
            plt.title(f'True: {class_names[test_labels[idx]]}\nPred: {class_names[y_test_pred[idx]]}')
            plt.axis('off')
            
            # Покажем вероятности
            probs = pipeline.predict_proba(X_test[idx:idx+1])[0]
            top3_idx = probs.argsort()[-3:][::-1]
            
            plt.subplot(1, 2, 2)
            plt.bar(range(3), [probs[i] for i in top3_idx])
            plt.xticks(range(3), [class_names[i] for i in top3_idx])
            plt.title('Top 3 Predictions')
            
            plt.tight_layout()
            plt.show()
    else:
        print("🎉 Ошибок нет!")


    # ============================================
    # Confusion Matrices
    # ============================================

    # После предсказаний, перед confusion matrix
    print("\n🔍 ДЕТАЛЬНАЯ ДИАГНОСТИКА:")
    print("="*40)

    # 1. Проверка распределения предсказаний
    print("\n1. Распределение истинных меток в val:")
    unique_val, counts_val = np.unique(val_labels, return_counts=True)
    for cls, cnt in zip(unique_val, counts_val):
        print(f"   {class_names[cls]}: {cnt}")

    print("\n2. Распределение ПРЕДСКАЗАННЫХ меток в val:")
    unique_pred, counts_pred = np.unique(y_val_pred, return_counts=True)
    for cls, cnt in zip(unique_pred, counts_pred):
        print(f"   {class_names[cls]}: {cnt}")

    # 3. Проверка, нет ли случайно всех предсказаний одинаковых
    if len(unique_pred) == 1:
        print("\n❌ КРИТИЧНО: Модель предсказывает ТОЛЬКО ОДИН класс!")
        print(f"   Все предсказания: {class_names[unique_pred[0]]}")
    else:
        print(f"\n✅ Модель предсказывает {len(unique_pred)} разных классов")

    # 4. Посмотрим на вероятности для нескольких примеров
    print("\n3. Примеры предсказаний с вероятностями:")
    for i in range(min(5, len(val_images))):
        probs = pipeline.predict_proba(X_val[i:i+1])[0]
        pred_idx = probs.argmax()
        confidence = probs[pred_idx]
        true_label = class_names[val_labels[i]]
        pred_label = class_names[pred_idx]
        
        print(f"   Пример {i+1}: Истинный={true_label}, Предсказанный={pred_label}, Уверенность={confidence:.3f}")
        
        # Если предсказание неверное, покажем топ-3
        if true_label != pred_label:
            top3_idx = probs.argsort()[-3:][::-1]
            print(f"      Топ-3: {[(class_names[idx], probs[idx]) for idx in top3_idx]}")


    print("\n📊 Plotting confusion matrices...")
    cm_val = plot_confusion_matrix(val_labels, y_val_pred, class_names, 
                                   'confusion_matrix_validation.png', 
                                   title='Validation Set')
    cm_test = plot_confusion_matrix(test_labels, y_test_pred, class_names, 
                                    'confusion_matrix_test.png', 
                                    title='Test Set')
    
    # ============================================
    # Misclassification analysis
    # ============================================
    plot_top_misclassifications(val_labels, y_val_pred, class_names, 
                               top_k=10, title="Validation Set")
    plot_top_misclassifications(test_labels, y_test_pred, class_names, 
                               top_k=10, title="Test Set")
    
    # ============================================
    # Save model with all accuracies
    # ============================================
    save_model(pipeline, class_names, train_accuracy, val_accuracy, test_accuracy, 
               filepath='hog_svm.pkl')
    

    distances, closest_pairs = visualize_class_distances(X_train, train_labels, class_names, pipeline)

    # distances, closest_pairs = visualize_class_distances(X_val, val_labels, class_names, pipeline)
    # distances, closest_pairs = visualize_class_distances(X_test, test_labels, class_names, pipeline)

    # Visualize HOG features for first few examples
    print("\n📸 Visualizing HOG features for sample images...")
    for i in range(min(3, len(train_images))):
        visualize_hog_features(train_images[i], hog, f'hog_visualization_sample_{i+1}.png')
    
    return pipeline, hog, class_names, train_accuracy, val_accuracy, test_accuracy

# ============================================
# 7. PREDICTION FUNCTION (using loaded model)
# ============================================

def predict_single_image(image_path, model_artifacts):
    
    pipeline = model_artifacts['pipeline']
    hog = model_artifacts['hog']
    class_names = model_artifacts['class_names']
    
    # Load and preprocess image using SAME transforms as validation/test
    transform = transforms.Compose([
        ExtractLetterWithMargin(margin=4, fill_white=True),
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    
    img_pil = Image.open(image_path).convert('L')
    img_transformed = transform(img_pil)
    img = np.array(img_transformed.squeeze())  # shape (64, 64)
    img = (img * 255).astype(np.uint8)  # Scale to 0-255
    
    # Display the preprocessed image
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title('Preprocessed Image for Prediction')
    plt.show()
    
    # Extract HOG features
    img_contiguous = np.ascontiguousarray(img)
    features = hog.compute(img_contiguous).flatten().reshape(1, -1)
    
    # Predict
    probs = pipeline.predict_proba(features)[0]
    pred_idx = probs.argmax()
    confidence = probs[pred_idx]
    
    predicted_letter = class_names[pred_idx]
    
    # Get top 5 predictions
    top5_idx = probs.argsort()[-5:][::-1]
    top5_predictions = [(class_names[i], probs[i]) for i in top5_idx]
    
    print(f"\n🔮 Prediction for {os.path.basename(image_path)}:")
    print(f"   Predicted: {predicted_letter}")
    print(f"   Confidence: {confidence*100:.2f}%")
    print(f"   Top 5 predictions:")
    for letter, prob in top5_predictions:
        print(f"      {letter}: {prob*100:.2f}%")
    
    return predicted_letter, confidence, top5_predictions

if __name__ == "__main__":
    try:
        # Set random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Train HOG+SVM with proper train/val/test split
        pipeline, hog, class_names, train_acc, val_acc, test_acc = main()
        
        artifacts = load_model('hog_svm.pkl')

        test_image = 'letter_a.jpeg'
        if os.path.exists(test_image):
            predict_single_image(test_image, artifacts)
        else:
            print(f"   Test image '{test_image}' not found")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()