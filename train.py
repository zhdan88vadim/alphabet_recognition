from models.model import AlphabetRecognizer
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import json
import yaml


from training.callbacks import EarlyStopping
from training.trainer import ModelTrainer
from data.augmentation import ExtractLetterWithMargin, SimpleThinOrThicken, Invert, RandomMissingPart, RandomBleed, AddRandomBlobs, RandomStrokeWidth, AddRandomBlackSpots

DATA_ROOT = "./dataset/test (копия)/"

def load_config(config_path='config/config.yaml'):
    """Загружает конфигурацию"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()

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
    
    print(f"\n Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['data']['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'], pin_memory=True)

    model = AlphabetRecognizer()
    model.to(device)
    
    # Подсчет параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 Model: AlphabetRecognizer")
    print(f"   Всего параметров: {total_params:,}")
    print(f"   Обучаемых: {trainable_params:,}")

    trainer = ModelTrainer(model, device, config)
    early_stopping = EarlyStopping(patience=config['training']['early_stopping_patience'])
    
    trainer.train(train_loader, val_loader, class_names, early_stopping)

if __name__ == "__main__":
    main()