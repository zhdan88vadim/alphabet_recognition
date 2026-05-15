import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
# Импорт ваших трансформаций
from data.augmentation import (
    ExtractLetterWithMargin, CenterDigitsTransform, SquarePad, 
    SimpleThinOrThicken, Invert, AddRandomBlobs, AddRandomBlackSpots,
    RandomStrokeWidth, RandomBleed, RandomMissingPart
)

# Загружаем изображение
img_path = "/mnt/ntfs/learn_ML/test_classes/Тестовое Python ML,CV/Тестовое_ML/тестовое_ml/dataset/test_unique_only/В/B_nu5kvt3g-13-1515.jpeg"  # укажите путь к вашей букве
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_pil = Image.fromarray(img_rgb)

# Параметры (как в вашем коде)
scale = 1
params = {
    'blob_size': (2, 4),
    'spot_size': (2, 4),
    'cut_size': (2, 4),
    'blur_radius': (0.5, 1.2),
    'stroke_width': (-1, 2),
    'translate': (0.1, 0.2),
    'shear': 15,
    'degrees': 10
}

# Список трансформаций по порядку

transforms_list = [
    ("0. Grayscale", transforms.Grayscale(num_output_channels=1)),
    ("1. ExtractLetterWithMargin", ExtractLetterWithMargin(margin=2, fill_white=True)),
    ("2. CenterDigitsTransform", CenterDigitsTransform(padding=2, fill_value=255)),
    ("3. SquarePad", SquarePad(fill_white=True)),
    ("4. SimpleThinOrThicken", SimpleThinOrThicken(p=1, strength='light', is_black_symbol_on_white_background=True)),
    ("5. Invert (1)", Invert()),
    ("6. Resize", transforms.Resize((64, 64))),
    ("7. RandomRotation", transforms.RandomRotation(20)),
    ("8. AddRandomBlobs (white)", AddRandomBlobs(p=1, num_blobs=(3, 5), blob_size=params['blob_size'], intensity=(250, 255))),
    ("9. AddRandomBlobs (black)", AddRandomBlobs(p=1, num_blobs=(3, 5), blob_size=params['blob_size'], intensity=(0, 5))),
    ("10. AddRandomBlackSpots", AddRandomBlackSpots(p=1, num_spots=(2, 5), spot_size=params['spot_size'])),
    ("11. RandomStrokeWidth", RandomStrokeWidth(p=1, thickness_range=params['stroke_width'])),
    ("12. RandomMissingPart", RandomMissingPart(p=1, cut_size=params['cut_size'])),
    ("13. RandomAffine", transforms.RandomAffine(degrees=params['degrees'], translate=params['translate'], shear=params['shear'])),
    ("14. Invert (2)", Invert()),
    ("15. ToTensor + Normalize", transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]))
]

# transforms_list = [
#      ("0. Grayscale", transforms.Grayscale(num_output_channels=1)),
#     ("1. ExtractLetterWithMargin", ExtractLetterWithMargin(margin=2, fill_white=True)),
#     ("2. CenterDigitsTransform", CenterDigitsTransform(padding=2, fill_value=255)),
#     ("3. SquarePad", SquarePad(fill_white=True)),
#     ("4. SimpleThinOrThicken", SimpleThinOrThicken(p=1, strength='light', min_thickness=1)),
#     ("5. Invert (1)", Invert()),
#     ("6. Resize", transforms.Resize((64, 64))),
#     ("7. RandomRotation", transforms.RandomRotation(20)),
#     ("8. AddRandomBlobs (white)", AddRandomBlobs(p=1, num_blobs=(3, 5), blob_size=params['blob_size'], intensity=(250, 255))),
#     ("9. AddRandomBlobs (black)", AddRandomBlobs(p=1, num_blobs=(3, 5), blob_size=params['blob_size'], intensity=(0, 5))),
#     ("10. AddRandomBlackSpots", AddRandomBlackSpots(p=1, num_spots=(2, 5), spot_size=params['spot_size'])),
#     ("11. RandomStrokeWidth", RandomStrokeWidth(p=1, thickness_range=params['stroke_width'])),
#     # ("12. RandomBleed", RandomBleed(p=1, blur_radius=params['blur_radius'])),
#     ("13. RandomMissingPart", RandomMissingPart(p=1, cut_size=params['cut_size'])),
#     ("14. RandomAffine", transforms.RandomAffine(degrees=params['degrees'], translate=params['translate'], shear=params['shear'])),
#     ("15. Invert (2)", Invert()),
#     ("16. Grayscale", transforms.Grayscale(num_output_channels=1)),
#     ("17. ToTensor + Normalize", transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]))
# ]

# Применяем и сохраняем результаты
current = img_pil
results = [("Original", current)]

print("Применяем трансформации...")
for name, transform_fn in transforms_list:
    try:
        current = transform_fn(current)
        results.append((name, current))
        print(f"✓ {name}")
    except Exception as e:
        print(f"✗ {name}: {e}")
        results.append((f"{name} (ERROR)", current))

# Отображаем всё в одной сетке
n = len(results)
cols = 4
rows = (n + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
axes = axes.flatten()

for i, (name, img) in enumerate(results):
    # Конвертируем тензор обратно в картинку для отображения
    if isinstance(img, torch.Tensor):
        img_vis = img.squeeze().cpu()
        img_vis = img_vis * 0.5 + 0.5
        img_vis = np.clip(img_vis.numpy(), 0, 1)
        axes[i].imshow(img_vis, cmap='gray')
    else:
        axes[i].imshow(img, cmap='gray' if img.mode == 'L' else None)
    
    axes[i].set_title(name, fontsize=8)
    axes[i].axis('off')

# Скрываем лишние
for i in range(n, len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('transformation_steps.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Сохранено: transformation_steps.png")