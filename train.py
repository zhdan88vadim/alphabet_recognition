from models.model import AlphabetRecognizer
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import yaml


from training.callbacks import EarlyStopping
from training.trainer import ModelTrainer
from data.augmentation import AdaptiveAugmentationBuilder

def load_config(config_path='config/config.yaml'):
    """Загружает конфигурацию"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Используем устройство: {device}")
    print(f"{'='*60}")

    aug_builder = AdaptiveAugmentationBuilder(base_size=config['data']['image_size'])
    
    train_transform = aug_builder.build_train_transform(
        (config['data']['image_size'], config['data']['image_size'])
    )
    val_transform = aug_builder.build_val_transform(
        (config['data']['image_size'], config['data']['image_size'])
    )

    print("📂 Загрузка датасета...")
    full_dataset = datasets.ImageFolder(
        root=config['data']['train_root'], 
        transform=train_transform
    )
    
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"📚 Найдено классов: {num_classes}")
    print(f"   {class_names}")
    
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