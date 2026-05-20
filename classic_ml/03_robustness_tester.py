# test_svm_robustness.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from collections import defaultdict
from pathlib import Path
import joblib
from PIL import Image
import torch
from torchvision import transforms

# Import same augmentations as training
from augmentation import CenterDigitsTransform, ExtractLetterWithMargin, Invert, SimpleThinOrThicken, SquarePad

# ============================================
# DISTORTION FUNCTIONS (same as CNN test)
# ============================================

def apply_rotation(image, angle, color):
    print(f"DEBUG: color = {color}, type = {type(color)}")
    h, w = image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=color)
    return rotated

def apply_translation(image, dx, dy, color):
    print(f"DEBUG: color = {color}, type = {type(color)}")
    h, w = image.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    translated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=color)
    return translated

def apply_scale(image, scale_factor, color):
    h, w = image.shape
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    scaled = cv2.resize(image, (new_w, new_h))
    if scale_factor > 1:
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        scaled = scaled[start_h:start_h+h, start_w:start_w+w]
    else:
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        scaled = cv2.copyMakeBorder(scaled, pad_h, h - new_h - pad_h,
                                   pad_w, w - new_w - pad_w,
                                   cv2.BORDER_CONSTANT, value=color)
    return scaled

# ============================================
# LOAD TEST DATA (same preprocessing as training)
# ============================================

def load_test_data(data_root):
    """Load test images with same preprocessing as validation/test set"""
    data_root = Path(data_root)
    class_names = sorted([d.name for d in data_root.iterdir() if d.is_dir()])
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    # Same transform as validation/test in training
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        ExtractLetterWithMargin(margin=2, fill_white=True),
        SquarePad(fill_white=True),
        transforms.Resize((64, 64)),
        Invert(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    X, y = [], []
    print("Loading test images...")
    for class_name in class_names:
        class_dir = data_root / class_name
        for img_path in class_dir.glob("*.*"):
            if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Apply same preprocessing
                img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                img_tensor = transform(img_pil)
                img_np = img_tensor.squeeze().cpu().numpy()
                # Denormalize to 0-255 for HOG
                img_np = ((img_np * 0.5 + 0.5) * 255).astype(np.uint8)
                
                X.append(img_np)
                y.append(class_to_idx[class_name])
    
    X = np.array(X)
    y = np.array(y)
    print(f"Loaded {len(X)} images, classes: {len(class_names)}")
    return X, y, class_names

# ============================================
# PREDICTION FUNCTIONS FOR SVM
# ============================================

def predict_single_svm(pipeline, hog, image_np):
    """Predict single image with SVM model"""
    # Ensure correct format
    if image_np.dtype != np.uint8:
        if image_np.max() <= 1:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
    
    # Resize if needed
    if image_np.shape != (64, 64):
        image_np = cv2.resize(image_np, (64, 64))
    
    # Extract HOG features
    img_contiguous = np.ascontiguousarray(image_np)
    features = hog.compute(img_contiguous).flatten().reshape(1, -1)
    
    # Predict
    probs = pipeline.predict_proba(features)[0]
    pred_idx = probs.argmax()
    confidence = probs[pred_idx] * 100
    
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3 = [(idx, probs[idx]*100) for idx in top3_idx]
    
    return pred_idx, confidence, top3

def predict_batch_svm(pipeline, hog, images_np, batch_size=64):
    """Predict batch of images with SVM model"""
    all_probs = []
    n = len(images_np)
    
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_imgs = images_np[start:end]
        
        batch_features = []
        for img in batch_imgs:
            if img.dtype != np.uint8:
                if img.max() <= 1:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            
            if img.shape != (64, 64):
                img = cv2.resize(img, (64, 64))
            
            img_contiguous = np.ascontiguousarray(img)
            features = hog.compute(img_contiguous).flatten()
            batch_features.append(features)
        
        batch_features = np.array(batch_features)
        probs = pipeline.predict_proba(batch_features)
        all_probs.append(probs)
    
    return np.vstack(all_probs)

# ============================================
# VISUALIZATION FUNCTIONS FOR SVM
# ============================================

def show_svm_distorted_predictions(pipeline, hog, X_test, y_test, class_names, n_samples=10):
    """Show predictions on distorted images"""
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    border_color = 0

    distortions = [
        ('Original', None),
        ('10°', lambda x: apply_rotation(x, 10, border_color)),
        ('-10°', lambda x: apply_rotation(x, -10, border_color)),
        ('20°', lambda x: apply_rotation(x, 20, border_color)),
        ('-20°', lambda x: apply_rotation(x, -20, border_color)),
        ('→5', lambda x: apply_translation(x, 5, 0, border_color)),
        ('↓5', lambda x: apply_translation(x, 0, 5, border_color)),
        ('→-10', lambda x: apply_translation(x, -10, 0, border_color)),
        ('↓-10', lambda x: apply_translation(x, 0, -10, border_color)),
        ('0.5x', lambda x: apply_scale(x, 0.5, border_color)),
        ('0.8x', lambda x: apply_scale(x, 0.8, border_color)),
        ('1.2x', lambda x: apply_scale(x, 1.2, border_color)),
    ]
    
    rows = len(indices)
    cols = len(distortions)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2.5))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for row, idx in enumerate(indices):
        true_label = class_names[y_test[idx]]
        original_img = X_test[idx].copy()
        
        for col, (dist_name, dist_func) in enumerate(distortions):
            ax = axes[row, col]
            img = original_img.copy()
            if dist_func:
                img = dist_func(img)
            
            pred_idx, conf, _ = predict_single_svm(pipeline, hog, img)
            pred_label = class_names[pred_idx]
            color = 'lime' if pred_idx == y_test[idx] else 'red'
            
            ax.imshow(img, cmap='gray')
            ax.text(0.5, 0.05, f'{pred_label}\n{conf:.0f}%', transform=ax.transAxes,
                    fontsize=8, color=color, fontweight='bold', ha='center', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            ax.axis('off')
            if row == 0:
                ax.set_title(dist_name, fontsize=8)
            if col == 0 and row == len(indices)//2:
                ax.set_ylabel('Distortions', fontsize=10, fontweight='bold')
            if col == 0:
                ax.text(-0.15, 0.5, true_label, transform=ax.transAxes,
                        fontsize=10, color='white', fontweight='bold', ha='center', va='center',
                        rotation=90, bbox=dict(boxstyle='round', facecolor='blue', alpha=0.8))
    
    plt.suptitle("SVM (HOG) recognition with distortions", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('svm_distorted_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: svm_distorted_predictions.png")

def show_svm_misclassified(pipeline, hog, X_test, y_test, class_names, n_samples=10):
    """Show misclassified examples"""
    print("Computing predictions for all images...")
    probs_all = predict_batch_svm(pipeline, hog, X_test)
    y_pred = np.argmax(probs_all, axis=1)
    
    mis_idx = np.where(y_pred != y_test)[0]
    if len(mis_idx) == 0:
        print("🎉 No errors!")
        return
    
    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Errors: {len(mis_idx)}/{len(y_test)} ({len(mis_idx)/len(y_test)*100:.2f}%)")
    
    error_pairs = defaultdict(int)
    for idx in mis_idx:
        error_pairs[(y_test[idx], y_pred[idx])] += 1
    
    print("\n🏆 TOP-3 ERRORS:")
    for i, ((true, pred), cnt) in enumerate(sorted(error_pairs.items(), key=lambda x: -x[1])[:3]):
        total = np.sum(y_test == true)
        print(f"   {i+1}. {class_names[true]} → {class_names[pred]}: {cnt} ({cnt/total*100:.1f}%)")
    
    n_show = min(n_samples, len(mis_idx))
    selected = np.random.choice(mis_idx, n_show, replace=False)
    cols = min(5, n_show)
    rows = (n_show + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
    axes = axes.flatten() if n_show > 1 else [axes]
    
    for i, idx in enumerate(selected):
        img = X_test[idx]
        true_lbl = class_names[y_test[idx]]
        pred_lbl = class_names[y_pred[idx]]
        probs = probs_all[idx]
        top3_idx = np.argsort(probs)[-3:][::-1]
        top3_text = ""
        for rank, cid in enumerate(top3_idx):
            mark = "✓" if cid == y_test[idx] else "✗" if rank == 0 else ""
            top3_text += f"{rank+1}. {class_names[cid]}: {probs[cid]*100:.1f}% {mark}\n"
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'True: {true_lbl}', fontsize=10)
        axes[i].text(1.2, -0.40, top3_text, transform=axes[i].transAxes, fontsize=8, family='monospace')
        axes[i].axis('off')
    
    for i in range(n_show, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f"SVM (HOG) misclassifications (total: {len(mis_idx)})", fontsize=14)
    plt.tight_layout()
    plt.savefig('svm_misclassified.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: svm_misclassified.png")

def show_svm_confusion_matrix(pipeline, hog, X_test, y_test, class_names):
    """Show confusion matrix"""
    print("Computing confusion matrix...")
    probs = predict_batch_svm(pipeline, hog, X_test)
    y_pred = np.argmax(probs, axis=1)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted class')
    plt.ylabel('True class')
    plt.title('SVM (HOG) Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('svm_confusion_matrix.png', dpi=150)
    plt.show()
    print("✅ Saved: svm_confusion_matrix.png")
    
    print("\n📊 Per-class statistics:")
    for i, name in enumerate(class_names):
        total = np.sum(y_test == i)
        correct = cm[i, i]
        acc = correct / total * 100 if total > 0 else 0
        print(f"   {name}: {correct}/{total} ({acc:.1f}%)")
    
    return cm

# ============================================
# MAIN FUNCTION
# ============================================

def main():
    # Paths
    data_root = "/mnt/ntfs/learn_ML/test_classes/Тестовое Python ML,CV/Тестовое_ML/тестовое_ml/dataset/classified_by_letter/"
    model_path = "hog_svm.pkl"
    
    # Load model
    print(f"Loading SVM model from {model_path}...")
    artifacts = joblib.load(model_path)
    pipeline = artifacts['pipeline']
    class_names = artifacts['class_names']
    
    # Recreate HOG descriptor
    hog_params = artifacts['hog_params']
    hog = cv2.HOGDescriptor(
        _winSize=hog_params['win_size'],
        _blockSize=hog_params['block_size'],
        _blockStride=hog_params['block_stride'],
        _cellSize=hog_params['cell_size'],
        _nbins=hog_params['nbins']
    )
    
    print(f"Model loaded. Classes: {len(class_names)}")
    print(f"Test accuracy from training: {artifacts.get('test_accuracy', 0)*100:.2f}%")
    
    # Load test data
    print("\n" + "="*60)
    X_test, y_test, data_class_names = load_test_data(data_root)
    
    # Verify classes match
    if set(data_class_names) != set(class_names):
        print("⚠️ Warning: Dataset and model classes differ!")
        print(f"   Dataset: {sorted(data_class_names)}")
        print(f"   Model: {sorted(class_names)}")
    
    print("\n" + "="*60)
    print("Testing SVM (HOG) Model Robustness")
    print("="*60)
    
    print("\n1. Recognition with distortions...")
    show_svm_distorted_predictions(pipeline, hog, X_test, y_test, class_names, n_samples=15)
    
    print("\n2. Misclassifications...")
    # show_svm_misclassified(pipeline, hog, X_test, y_test, class_names, n_samples=40)
    
    print("\n3. Confusion matrix...")
    # show_svm_confusion_matrix(pipeline, hog, X_test, y_test, class_names)
    
    print("\n" + "="*60)
    print("✅ Done!")
    print("Saved files:")
    print("   • svm_distorted_predictions.png")
    print("   • svm_misclassified.png")
    print("   • svm_confusion_matrix.png")
    print("="*60)

if __name__ == "__main__":
    np.random.seed(42)
    main()