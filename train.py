from models.model import AlphabetRecognizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import json
import copy


from training.callbacks import EarlyStopping
from transforms_helper import ExtractLetterWithMargin, SimpleThinOrThicken, Invert, RandomMissingPart, RandomBleed, AddRandomBlobs, RandomStrokeWidth, AddRandomBlackSpots

DATA_ROOT = "./dataset/test (копия)/"

def main():
    global val_loader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Используем устройство: {device}")
    print(f"{'='*60}")

    train_transform = transforms.Compose([
        ExtractLetterWithMargin(margin=2, fill_white=True),
        transforms.Resize((64, 64)),
        # transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        Invert(),
            
        AddRandomBlobs(p=0.5, num_blobs=(3, 5), blob_size=(2, 4), intensity=(250, 255)),
        AddRandomBlobs(p=0.5, num_blobs=(3, 5), blob_size=(2, 4), intensity=(0, 5)),
        AddRandomBlackSpots(p=0.5, num_spots=(2, 5), spot_size=(2, 4)),
        
        RandomStrokeWidth(p=0.5, thickness_range=(-1, 2)),  # изменение толщины
        RandomBleed(p=0.5, blur_radius=(0.5, 1.2)),  # растекание
        RandomMissingPart(p=0.5, cut_size=(2, 4)),  # отсутствующая часть

        # transforms.RandomRotation(degrees=15), 
        transforms.RandomAffine(
            degrees=10, 
            translate=(0.1, 0.2), 
            shear=15
        ),
        # ElasticTransform(alpha=8, sigma=2, p=1),
        
        # # Добавляем больше аугментаций
        # transforms.RandomPerspective(distortion_scale=0.4, p=0.5),     
        # transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        SimpleThinOrThicken(p=1, strength='medium', min_thickness=1),
        
        Invert(),   
        # transforms.GaussianBlur(kernel_size=3, sigma=(0.4, 0.9)),
        
        # Конвертируем в 3 канала перед ToTensor
        transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        # AddGaussianNoise(), 
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    val_transform = transforms.Compose([
        ExtractLetterWithMargin(margin=2, fill_white=True),
        transforms.Resize((64, 64)),
        Invert(),
        SimpleThinOrThicken(p=1, strength='medium', min_thickness=1),
        Invert(),
        # transforms.Lambda(lambda x: 255 - np.array(x) if isinstance(x, Image.Image) else 255 - x),
        # transforms.ToPILImage(),  # обратно в PIL        
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    print("📂 Загрузка датасета...")
    full_dataset = datasets.ImageFolder(root=DATA_ROOT, transform=train_transform)
    
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"📚 Найдено классов: {num_classes}")
    print(f"   {class_names}")

    with open("class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False)
    
    # Подсчет количества изображений на класс
    class_counts = {}
    for idx, (_, label) in enumerate(full_dataset.samples):
        class_name = class_names[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print(f"\n📊 Распределение по классам:")
    for name, count in sorted(class_counts.items()):
        print(f"   {name}: {count} изображений")
    

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")

    # Меняем трансформацию для валидации
    val_dataset.dataset.transform = val_transform
    
    print(f"\n📦 Train samples: {len(train_dataset)}")
    print(f"📦 Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    model = AlphabetRecognizer()
    model.to(device)
    
    # Подсчет параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 Model: AlphabetRecognizer")
    print(f"   Всего параметров: {total_params:,}")
    print(f"   Обучаемых: {trainable_params:,}")

    # # Создаем веса для классов
    # class_weights = torch.ones(num_classes)
    # # Увеличиваем вес для В и Р
    # class_weights[class_names.index('В')] = 1.5
    # class_weights[class_names.index('Р')] = 1.5

    # --- Оптимизатор с weight decay ---
    # criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    criterion = nn.CrossEntropyLoss()
    
    # criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0005, # 0.0001
        weight_decay=0.02  # L2 регуляризация
    )
    
    # Scheduler: уменьшаем lr при затухании валидации
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=2, 
    )
    
    early_stopping = EarlyStopping(patience=3, min_delta=0.001)
    

    epochs = 20
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    train_losses = []
    val_accs = []
    
    print(f"\nНачинаем обучение на {epochs} эпохах...")
    print(f"{'='*60}")
    
    for epoch in range(epochs):
        # ---------- Training ----------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Train Epoch {epoch+1}")
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping для стабильности
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            train_acc = 100 * correct / total
            loop.set_postfix(loss=loss.item(), acc=f"{train_acc:.1f}%")
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        
        # ---------- Validation ----------
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        val_accs.append(val_acc)
        
        # ---------- Отчет ----------
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Разрыв между train и val (индикатор переобучения)
        gap = train_acc - val_acc
        if gap > 5:
            print(f"  ⚠️ Разрыв Train-Val: {gap:.2f}% (возможно переобучение)")
        
        # ---------- Scheduler ----------
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # ---------- Сохранение лучшей модели ----------
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names,
            }, 'best_alphabet_model.pth')
            print(f"  Сохранена лучшая модель (Acc: {val_acc:.2f}%)")
        
        # ---------- Early Stopping ----------
        if early_stopping(val_acc):
            print(f"\n🛑 Early stopping triggered на эпохе {epoch+1}")
            break
        
        # Дополнительная проверка: если точность валидации падает 3 эпохи подряд
        if len(val_accs) > 3:
            if val_accs[-1] < val_accs[-2] < val_accs[-3]:
                print(f"  ⚠️ Val accuracy падает 3 эпохи подряд!")
    
    # Load better model
    model.load_state_dict(best_model_wts)
 
    print(f"\n{'='*60}")
    print(f"Обучение завершено!")
    print(f"Лучшая точность на валидации: {best_acc:.2f}%")
    print(f"Модель сохранена в 'best_alphabet_model.pth'")
    print(f"Маппинг классов в 'class_mapping.json'")
    print(f"{'='*60}")
    
    with open("training_history.json", "w", encoding="utf-8") as f:
        json.dump({
            'train_losses': train_losses,
            'val_accs': val_accs,
            'best_acc': best_acc,
            'num_classes': num_classes,
            'class_names': class_names
        }, f)

if __name__ == "__main__":
    main()