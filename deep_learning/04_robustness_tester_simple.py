import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import os

# Импорт ваших трансформаций
from augmentation import CenterDigitsTransform, SquarePad

DATA_ROOT = "/media/vadim/1TB_SSD/my_github/alphabet_recognition/dataset/test (копия)"

# ============================================
# ПРОСТЫЕ ФУНКЦИИ ДЛЯ ИСКАЖЕНИЙ
# ============================================

def apply_rotation(image, angle):
    """Поворот изображения"""
    h, w = image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0) # 255
    return rotated

def apply_translation(image, dx, dy):
    """Сдвиг изображения"""
    h, w = image.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    translated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0) # 255
    return translated

def apply_scale(image, scale_factor):
    """Масштабирование"""
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
                                   cv2.BORDER_CONSTANT, value=0) # 255
    return scaled

def extract_hog_features_simple(images, hog, image_size=(64, 64)):
    """Простое извлечение HOG признаков"""
    features_list = []
    for img in images:
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if img.shape != image_size:
            img = cv2.resize(img, image_size)
        img = np.ascontiguousarray(img)
        feat = hog.compute(img).flatten()
        features_list.append(feat)
    return np.vstack(features_list)

def create_hog_descriptor():
    """Создание HOG дескриптора"""
    return cv2.HOGDescriptor(
        _winSize=(64, 64),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )

def binarize_image(image, threshold=127):
    """Преобразование изображения в бинарное (черное/белое)"""
    # Если изображение не в uint8, конвертируем
    if image.dtype != np.uint8:
        if image.max() <= 1:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Бинаризация: всё что выше threshold становится 255, ниже - 0
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary

# ============================================
# ЗАГРУЗКА ДАННЫХ
# ============================================

def load_data(data_root):
    """Загрузка данных в numpy"""
    base_transform = transforms.Compose([
        CenterDigitsTransform(padding=2, fill_value=0),
        SquarePad(fill_white=False),
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.ImageFolder(root=data_root, transform=base_transform)
    class_names = dataset.classes
    
    all_images = []
    all_labels = []
    
    print("Загрузка данных...")
    for i in tqdm(range(len(dataset))):
        img_tensor, label = dataset[i]
        img_np = img_tensor.squeeze().cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)

        _, img_np = cv2.threshold(img_np, 127, 255, cv2.THRESH_BINARY)

        all_images.append(img_np)
        all_labels.append(label)
    
    return np.stack(all_images), np.array(all_labels), class_names

# ============================================
# ВИЗУАЛИЗАЦИЯ
# ============================================

def show_dataset_samples(X, y, class_names, samples_per_class=3):
    """Показать примеры из датасета"""
    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes, samples_per_class, figsize=(samples_per_class * 2, n_classes * 2))
    
    for class_idx in range(n_classes):
        class_indices = np.where(y == class_idx)[0]
        selected = np.random.choice(class_indices, min(samples_per_class, len(class_indices)), replace=False)
        
        for sample_idx, img_idx in enumerate(selected):
            ax = axes[class_idx, sample_idx] if n_classes > 1 else axes[sample_idx]
            ax.imshow(X[img_idx], cmap='gray')
            ax.set_title(class_names[class_idx], fontsize=10)
            ax.axis('off')
    
    plt.suptitle("Примеры из датасета", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('dataset_samples.png', dpi=150)
    plt.show()
    print("✅ Сохранено: dataset_samples.png")

def show_distorted_predictions(model, hog, X_test, y_test, class_names, n_samples):
    """Показать оригинал и искаженные версии с предсказаниями (текст на изображении)"""
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    distortions = [
        ('Оригинал', None),
        ('10°', lambda x: apply_rotation(x, 10)),
        ('-10°', lambda x: apply_rotation(x, -10)),
        ('20°', lambda x: apply_rotation(x, 20)),
        ('-20°', lambda x: apply_rotation(x, -20)),
        ('→5', lambda x: apply_translation(x, 5, 0)),
        ('↓5', lambda x: apply_translation(x, 0, 5)),
        ('→-10', lambda x: apply_translation(x, -10, 0)),
        ('↓-10', lambda x: apply_translation(x, 0, -10)),
        ('0.5x', lambda x: apply_scale(x, 0.5)),
        ('0.8x', lambda x: apply_scale(x, 0.8)),
        ('1.2x', lambda x: apply_scale(x, 1.2)),
    ]
    
    fig, axes = plt.subplots(len(indices), len(distortions), 
                             figsize=(len(distortions) * 2, len(indices) * 2.5))
    
    if len(indices) == 1:
        axes = axes.reshape(1, -1)
    
    for row, idx in enumerate(indices):
        true_label = class_names[y_test[idx]]
        
        for col, (dist_name, dist_func) in enumerate(distortions):
            if dist_func is None:
                img = X_test[idx].copy()
            else:
                img = dist_func(X_test[idx].copy())
            
            features = extract_hog_features_simple([img], hog)
            pred = model.predict(features)[0]
            pred_label = class_names[pred]
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features)[0][pred] * 100
                text = f'{pred_label}\n{proba:.0f}%'
            else:
                text = pred_label
            
            color = 'lime' if pred == y_test[idx] else 'red'
            
            ax = axes[row, col]
            ax.imshow(img, cmap='gray')
            
            # Добавляем текст поверх изображения
            ax.text(0.5, 0.05, text, transform=ax.transAxes,
                   fontsize=10, color=color, fontweight='bold',
                   ha='center', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            
            # Заголовок сверху
            if row == 0:
                ax.set_title(dist_name, fontsize=9, fontweight='bold')
            
            # Подпись слева
            if col == 0:
                ax.text(-0.15, 0.5, true_label, transform=ax.transAxes,
                       fontsize=11, color='white', fontweight='bold',
                       ha='center', va='center', rotation=90,
                       bbox=dict(boxstyle='round', facecolor='blue', alpha=0.8))
            
            ax.axis('off')
    
    plt.suptitle("Распознавание с искажениями", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('distorted_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Сохранено: distorted_predictions.png")

def show_misclassified(model, hog, X_test, y_test, class_names, n_samples=10):
    """Показать ошибочно распознанные изображения с вероятностями для ТОП-3 классов"""
    # Получаем предсказания и вероятности
    X_test_hog = extract_hog_features_simple(X_test, hog)
    test_proba = model.predict_proba(X_test_hog)
    y_pred = np.argmax(test_proba, axis=1)
    
    # Находим ошибки
    misclassified_idx = np.where(y_pred != y_test)[0]
    
    if len(misclassified_idx) == 0:
        print("🎉 Нет ошибок на тестовой выборке!")
        return
    
    print(f"Ошибок: {len(misclassified_idx)}/{len(y_test)} ({len(misclassified_idx)/len(y_test)*100:.2f}%)")
    
    # ТОП-3 самых частых ошибок
    from collections import defaultdict
    error_pairs = defaultdict(int)
    for idx in misclassified_idx:
        error_pairs[(y_test[idx], y_pred[idx])] += 1
    
    print("\n🏆 ТОП-3 ОШИБКИ:")
    for i, ((true, pred), count) in enumerate(sorted(error_pairs.items(), key=lambda x: -x[1])[:3]):
        total = np.sum(y_test == true)
        print(f"   {i+1}. {class_names[true]} → {class_names[pred]}: {count} ({count/total*100:.1f}%)")
    
    # Показываем примеры ошибок
    n_show = min(n_samples, len(misclassified_idx))
    selected_idx = np.random.choice(misclassified_idx, n_show, replace=False)
    
    cols = min(5, n_show)
    rows = (n_show + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten() if n_show > 1 else [axes]
    
    for i, idx in enumerate(selected_idx):
        img = X_test[idx]
        true_label = class_names[y_test[idx]]
        pred_label = class_names[y_pred[idx]]
        
        # Получаем ТОП-3 предсказания с вероятностями
        top3_idx = np.argsort(test_proba[idx])[-3:][::-1]
        
        # Формируем текст с ТОП-3 вероятностями
        top3_text = ""
        for rank, class_idx in enumerate(top3_idx):
            proba = test_proba[idx][class_idx] * 100
            mark = "✓" if class_idx == y_test[idx] else "✗" if rank == 0 else ""
            top3_text += f"{rank+1}. {class_names[class_idx]}: {proba:.1f}% {mark}\n"
        
        # Отображаем изображение
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'True: {true_label}', fontsize=10, fontweight='bold')
        axes[i].text(0.5, -0.15, top3_text, transform=axes[i].transAxes, 
                    fontsize=8, verticalalignment='top', family='monospace')
        axes[i].axis('off')
    
    for i in range(n_show, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f"Ошибочные предсказания (всего: {len(misclassified_idx)})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('misclassified.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n✅ Сохранено: misclassified.png")
    
    # Дополнительно выводим в консоль детали по каждому примеру
    print("\n📊 ДЕТАЛИ ОШИБОК В КОНСОЛИ:")
    print("="*60)
    for i, idx in enumerate(selected_idx[:5]):
        print(f"\n{i+1}. Индекс {idx}:")
        print(f"   Истинная: {class_names[y_test[idx]]}")
        print(f"   Предсказано: {class_names[y_pred[idx]]}")
        print(f"   ТОП-3 вероятности:")
        top3_idx = np.argsort(test_proba[idx])[-3:][::-1]
        for rank, class_idx in enumerate(top3_idx):
            proba = test_proba[idx][class_idx] * 100
            mark = "← ОШИБКА" if rank == 0 and class_idx != y_test[idx] else "← ВЕРНО" if class_idx == y_test[idx] else ""
            print(f"      {rank+1}. {class_names[class_idx]}: {proba:.1f}% {mark}")

def show_confusion_matrix(model, hog, X_test, y_test, class_names):
    """Показать матрицу ошибок"""
    X_test_hog = extract_hog_features_simple(X_test, hog)
    y_pred = model.predict(X_test_hog)
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Нормализованная матрица
    # cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='.1f', cmap='YlOrRd',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Процент (%)'})
    
    plt.xlabel('Предсказанный класс', fontsize=12)
    plt.ylabel('Истинный класс', fontsize=12)
    plt.title('Матрица ошибок (%)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()
    print("✅ Сохранено: confusion_matrix.png")
    
    # Вывод статистики ошибок
    print("\n📊 Статистика ошибок по классам:")
    for i, class_name in enumerate(class_names):
        total = np.sum(y_test == i)
        correct = cm[i, i]
        acc = correct / total * 100
        print(f"   {class_name}: {correct}/{total} ({acc:.1f}%)")

# ============================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================

def main():
    print("="*60)
    print("🔍 ВИЗУАЛИЗАЦИЯ РАСПОЗНАВАНИЯ ЦИФР")
    print("="*60)
    
    # 1. Загрузка данных
    X, y, class_names = load_data(DATA_ROOT)
    print(f"Загружено: {len(X)} изображений, {len(class_names)} классов")
    
    # 2. Загрузка модели (выберите одну)
    model_files = ['hog_rf_model.pkl']
    # model_files = ['hog_svm_model.pkl']
    model_path = None
    
    for f in model_files:
        if os.path.exists(f):
            model_path = f
            break
    
    if model_path is None:
        print("❌ Модель не найдена!")
        return
    
    print(f"Загрузка модели: {model_path}")
    artifacts = joblib.load(model_path)
    model = artifacts['pipeline']
    class_names = artifacts.get('class_names', class_names)
    
    # 3. Создание HOG дескриптора
    hog = create_hog_descriptor()
    
    # 4. Визуализация
    print("\n" + "="*60)
    print("📊 ВИЗУАЛИЗАЦИЯ")
    print("="*60)
    
    # Примеры из датасета
    print("\n1. Примеры из датасета...")
    # show_dataset_samples(X, y, class_names, samples_per_class=3)
    
    # Распознавание с искажениями
    print("\n2. Тестирование с искажениями...")
    show_distorted_predictions(model, hog, X, y, class_names, n_samples=15)
    
    # Ошибочные предсказания
    print("\n3. Поиск ошибок...")
    show_misclassified(model, hog, X, y, class_names, n_samples=40)
    
    # Матрица ошибок
    print("\n4. Матрица ошибок...")
    show_confusion_matrix(model, hog, X, y, class_names)
    
    print("\n" + "="*60)
    print("✅ ГОТОВО!")
    print("Созданные файлы:")
    print("   • dataset_samples.png - примеры из датасета")
    print("   • distorted_predictions.png - распознавание с искажениями")
    print("   • misclassified.png - ошибочные предсказания")
    print("   • confusion_matrix.png - матрица ошибок")
    print("="*60)

if __name__ == "__main__":
    np.random.seed(42)
    main()