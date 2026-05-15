import cv2
from data.augmentation import CenterDigitsTransform, ExtractLetterWithMargin, Invert, SimpleThinOrThicken, SquarePad
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
from torchvision import transforms
from PIL import Image
import json
from collections import defaultdict
from pathlib import Path

# Импорт вашей модели (убедитесь, что путь правильный)
from models.model import AlphabetRecognizer

# ============================================
# ФУНКЦИИ ИСКАЖЕНИЙ (как в SVM коде)
# ============================================

def apply_rotation(image, angle):
    h, w = image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return rotated

def apply_translation(image, dx, dy):
    h, w = image.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    translated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return translated

def apply_scale(image, scale_factor):
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
                                   cv2.BORDER_CONSTANT, value=255)
    return scaled

# ============================================
# ЗАГРУЗКА ТЕСТОВЫХ ДАННЫХ
# ============================================

def load_data(data_root):
    data_root = Path(data_root)
    class_names = sorted([d.name for d in data_root.iterdir() if d.is_dir()])
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    X, y = [], []
    print("Загрузка тестовых изображений...")
    for class_name in class_names:
        class_dir = data_root / class_name
        for img_path in class_dir.glob("*.*"):
            if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                img = cv2.resize(img, (64, 64))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                # _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
                X.append(binary)
                y.append(class_to_idx[class_name])
    
    X = np.array(X)
    y = np.array(y)
    print(f"Загружено {len(X)} изображений, классов: {len(class_names)}")
    return X, y, class_names

# ============================================
# ЗАГРУЗКА CNN МОДЕЛИ
# ============================================

def load_cnn_model(model_path, mapping_path, device):
    with open(mapping_path, 'r', encoding='utf-8') as f:
        class_names = json.load(f)
    if isinstance(class_names, dict):
        class_names = list(class_names.values())
    
    checkpoint = torch.load(model_path, map_location=device)
    model = AlphabetRecognizer()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"CNN модель загружена: {model_path}, классов: {len(class_names)}")
    return model, class_names

def get_cnn_transform():
    return transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),
        ExtractLetterWithMargin(margin=2, fill_white=True),
        # CenterDigitsTransform(padding=10, fill_value=255),
        SquarePad(fill_white=True),

        # HERE image already 64 px
        # need make less strength
        # SimpleThinOrThicken(p=1, strength='light', is_black_symbol_on_white_background=True),
        
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])

        # transforms.Grayscale(num_output_channels=1),
        # ExtractLetterWithMargin(margin=2, fill_white=True),
        # # Invert(),
        # CenterDigitsTransform(padding=2, fill_value=0),
        # SquarePad(fill_white=False),
        # transforms.Resize((64, 64)),
        # SimpleThinOrThicken(p=1, strength='medium', is_black_symbol_on_white_backround=True),
        # Invert(),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])        
        # ExtractLetterWithMargin(margin=2, fill_white=True),
        # # Invert(),
        # CenterDigitsTransform(padding=2, fill_value=255),
        # SquarePad(fill_white=True),
        # transforms.Resize((64, 64)),
        # # Invert(),
        # # SimpleThinOrThicken(p=1, strength='light', min_thickness=1),
        # # transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
        # transforms.Grayscale(num_output_channels=1),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def visualize_augmented_samples(model, transform, device, X_test, y_test, class_names, num_samples=5):
    """
    Показывает в отдельном окне несколько случайных изображений
    после полного пайплайна аугментаций (то, что видит модель)
    """
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 3, 3))
    if num_samples == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        img = X_test[idx].copy()
        # Применяем тот же пайплайн, что и перед подачей в модель
        img_tensor = preprocess_for_cnn(img, transform).to(device)
        # Денормализуем для отображения
        img_vis = img_tensor.squeeze().cpu() * 0.5 + 0.5
        img_vis = np.clip(img_vis.numpy(), 0, 1) * 255
        img_vis = img_vis.astype(np.uint8)
        
        # Получаем предсказание
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx] * 100
        
        true_label = class_names[y_test[idx]]
        pred_label = class_names[pred_idx]
        color = 'green' if pred_idx == y_test[idx] else 'red'
        
        axes[i].imshow(img_vis, cmap='gray')
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%', 
                          fontsize=9, color=color)
        axes[i].axis('off')
    
    plt.suptitle("Примеры изображений после аугментаций (вход модели)", fontsize=12)
    plt.tight_layout()
    plt.show(block=True)  # блокирует выполнение, пока окно не закроют
    print("✅ Окно с примерами закрыто, продолжаем...")

# def preprocess_for_cnn(image_np, transform):
#     # Инвертируем, чтобы буквы были белыми на чёрном (как ожидает модель)
#     inverted = cv2.bitwise_not(image_np)
#     img_pil = Image.fromarray(inverted)
#     return transform(img_pil).unsqueeze(0)
#     # inverted = cv2.bitwise_not(image_np)
#     # img_pil = Image.fromarray(inverted)
#     # return transform(img_pil).unsqueeze(0)

def preprocess_for_cnn(image_np, transform):
    # Определяем, светлый ли фон (белый)
    # Для бинарного изображения: если среднее > 127, то преобладает белый цвет
    # if np.mean(image_np) > 127:
    #     image_np = cv2.bitwise_not(image_np)
    img_pil = Image.fromarray(image_np)
    return transform(img_pil).unsqueeze(0)

# ============================================
# ПРЕДСКАЗАНИЯ С БАТЧАМИ (исправление OOM)
# ============================================

def predict_batch(model, images_np, transform, device, batch_size=64):
    """Предсказание для всех изображений с разбиением на батчи"""
    model.eval()
    all_probs = []
    n = len(images_np)
    
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_imgs = images_np[start:end]
            
            tensors = [preprocess_for_cnn(img, transform).to(device) for img in batch_imgs]
            batch_tensor = torch.cat(tensors, dim=0)
            outputs = model(batch_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)
    
    return np.vstack(all_probs)

def predict_single(model, image_np, transform, device):
    probs = predict_batch(model, [image_np], transform, device, batch_size=1)[0]
    pred_idx = np.argmax(probs)
    confidence = probs[pred_idx] * 100
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3 = [(idx, probs[idx]*100) for idx in top3_idx]
    return pred_idx, confidence, top3

# ============================================
# ВИЗУАЛИЗАЦИИ
# ============================================

def show_distorted_predictions(model, transform, device, X_test, y_test, class_names, n_samples=15):
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
    
    rows = len(indices)
    cols = len(distortions) + 1  # +1 для колонки "После аугментаций (оригинал)"
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2.5))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for row, idx in enumerate(indices):
        true_label = class_names[y_test[idx]]
        original_img = X_test[idx].copy()
        
        # Колонки с искажениями
        for col, (dist_name, dist_func) in enumerate(distortions):
            ax = axes[row, col]
            img = original_img.copy()
            if dist_func:
                img = dist_func(img)
            
            # Предсказание для искажённого изображения
            pred_idx, conf, _ = predict_single(model, img, transform, device)
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
                ax.set_ylabel('Искажения', fontsize=10, fontweight='bold')
            if col == 0:
                ax.text(-0.15, 0.5, true_label, transform=ax.transAxes,
                        fontsize=10, color='white', fontweight='bold', ha='center', va='center',
                        rotation=90, bbox=dict(boxstyle='round', facecolor='blue', alpha=0.8))
        
        # Последняя колонка: оригинал после полных аугментаций (вход модели)
        ax_aug = axes[row, -1]
        img_aug = apply_transform_for_display(original_img, transform)
        # Предсказание для аугментированного оригинала
        pred_idx, conf, _ = predict_single(model, original_img, transform, device)
        pred_label = class_names[pred_idx]
        color = 'lime' if pred_idx == y_test[idx] else 'red'
        
        ax_aug.imshow(img_aug, cmap='gray')
        ax_aug.text(0.5, 0.05, f'{pred_label}\n{conf:.0f}%', transform=ax_aug.transAxes,
                    fontsize=8, color=color, fontweight='bold', ha='center', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        ax_aug.axis('off')
        if row == 0:
            ax_aug.set_title('После аугментаций\n(оригинал)', fontsize=8)
        if row == len(indices)//2:
            ax_aug.set_ylabel('Вход модели', fontsize=10, fontweight='bold')
    
    plt.suptitle("Распознавание CNN с искажениями + вход модели для оригинала", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cnn_distorted_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Сохранено: cnn_distorted_predictions.png")



def show_misclassified(model, transform, device, X_test, y_test, class_names, n_samples=10, batch_size=64):
    print("Вычисление предсказаний для всех изображений...")
    probs_all = predict_batch(model, X_test, transform, device, batch_size)
    y_pred = np.argmax(probs_all, axis=1)
    
    mis_idx = np.where(y_pred != y_test)[0]
    if len(mis_idx) == 0:
        print("🎉 Нет ошибок!")
        return
    
    print(f"Ошибок: {len(mis_idx)}/{len(y_test)} ({len(mis_idx)/len(y_test)*100:.2f}%)")
    
    # ТОП-3 частых ошибок
    error_pairs = defaultdict(int)
    for idx in mis_idx:
        error_pairs[(y_test[idx], y_pred[idx])] += 1
    print("\n🏆 ТОП-3 ОШИБКИ:")
    for i, ((true, pred), cnt) in enumerate(sorted(error_pairs.items(), key=lambda x: -x[1])[:3]):
        total = np.sum(y_test == true)
        print(f"   {i+1}. {class_names[true]} → {class_names[pred]}: {cnt} ({cnt/total*100:.1f}%)")
    
    # Показываем примеры
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
    
    plt.suptitle(f"Ошибочные предсказания CNN (всего: {len(mis_idx)})", fontsize=14)
    plt.tight_layout()
    plt.savefig('cnn_misclassified.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Сохранено: cnn_misclassified.png")
    
    # Детали в консоли
    print("\n📊 ДЕТАЛИ ОШИБОК В КОНСОЛИ:")
    print("="*60)
    for i, idx in enumerate(selected[:5]):
        print(f"\n{i+1}. Индекс {idx}:")
        print(f"   Истинная: {class_names[y_test[idx]]}")
        print(f"   Предсказано: {class_names[y_pred[idx]]}")
        probs = probs_all[idx]
        top3_idx = np.argsort(probs)[-3:][::-1]
        for rank, cid in enumerate(top3_idx):
            mark = "← ОШИБКА" if rank == 0 and cid != y_test[idx] else "← ВЕРНО" if cid == y_test[idx] else ""
            print(f"      {rank+1}. {class_names[cid]}: {probs[cid]*100:.1f}% {mark}")

def apply_transform_for_display(img_np, transform):
    """
    Применяет полный пайплайн трансформаций (кроме Normalize) к numpy-изображению
    и возвращает uint8 массив для визуализации.
    """
    # Применяем все трансформации (включая ToTensor и Normalize)
    img_pil = Image.fromarray(img_np)
    tensor = transform(img_pil).unsqueeze(0)  # [1,1,64,64], нормализованный
    
    # Денормализация: mean=0.5, std=0.5 -> (tensor * 0.5 + 0.5)
    tensor_vis = tensor * 0.5 + 0.5
    # Обрезаем значения до [0,1] и конвертируем в numpy
    img_vis = tensor_vis.squeeze().cpu().numpy()
    img_vis = np.clip(img_vis, 0, 1)
    img_vis = (img_vis * 255).astype(np.uint8)
    return img_vis

def show_confusion_matrix(model, transform, device, X_test, y_test, class_names, batch_size=64):
    print("Вычисление матрицы ошибок...")
    probs = predict_batch(model, X_test, transform, device, batch_size)
    y_pred = np.argmax(probs, axis=1)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Количество'})
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.title('Матрица ошибок CNN', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cnn_confusion_matrix.png', dpi=150)
    plt.show()
    print("✅ Сохранено: cnn_confusion_matrix.png")
    
    print("\n📊 Статистика по классам:")
    for i, name in enumerate(class_names):
        total = np.sum(y_test == i)
        correct = cm[i, i]
        acc = correct / total * 100 if total > 0 else 0
        print(f"   {name}: {correct}/{total} ({acc:.1f}%)")

# ============================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================

def main():
    # Укажите свои пути
    # data_root = "/media/vadim/1TB_SSD/my_github/alphabet_recognition/dataset/test (копия)/"
    # data_root = "/mnt/ntfs/learn_ML/test_classes/Тестовое Python ML,CV/Тестовое_ML/тестовое_ml/dataset/classified_by_letter/"
    data_root = "/mnt/ntfs/learn_ML/test_classes/Тестовое Python ML,CV/Тестовое_ML/тестовое_ml/dataset/test_unique_only/"
    
    model_path = "best_alphabet_model.pth"
    mapping_path = "class_mapping.json"
    batch_size = 64  # уменьшите до 32, если всё ещё не хватает памяти
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    # Загрузка данных
    X, y, data_class_names = load_data(data_root)
    
    # Загрузка модели
    model, cnn_class_names = load_cnn_model(model_path, mapping_path, device)
    
    # Сверка классов (если не совпадают, используем классы модели)
    if set(data_class_names) != set(cnn_class_names):
        print("⚠️ Внимание: классы в датасете и модели различаются!")
        print(f"   Датасет: {sorted(data_class_names)}")
        print(f"   Модель: {sorted(cnn_class_names)}")
        # Здесь можно добавить переиндексацию, если нужно.
        # По умолчанию используем классы модели.
    class_names = cnn_class_names
    # class_names = data_class_names
    
    transform = get_cnn_transform()

    visualize_augmented_samples(model, transform, device, X, y, class_names, num_samples=5)
    
    print("\n" + "="*60)
    print("Тестирование устойчивости CNN модели")
    print("="*60)
    
    print("\n1. Распознавание с искажениями...")
    show_distorted_predictions(model, transform, device, X, y, class_names, n_samples=15)
    
    print("\n2. Ошибочные предсказания...")
    show_misclassified(model, transform, device, X, y, class_names, n_samples=40, batch_size=batch_size)
    
    print("\n3. Матрица ошибок...")
    show_confusion_matrix(model, transform, device, X, y, class_names, batch_size=batch_size)
    
    print("\n" + "="*60)
    print("✅ Готово!")
    print("Сохранены файлы:")
    print("   • cnn_distorted_predictions.png")
    print("   • cnn_misclassified.png")
    print("   • cnn_confusion_matrix.png")
    print("="*60)

if __name__ == "__main__":
    np.random.seed(42)
    main()