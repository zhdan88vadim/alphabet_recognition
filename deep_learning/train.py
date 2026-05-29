from models.model import AlphabetRecognizer
from tensorboard_utils.visualizer import log_transformed_images, log_transformed_images_from_dataloader
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
import yaml
import matplotlib.pyplot as plt

from training.callbacks import EarlyStopping
from training.trainer import ModelTrainer
from data.augmentation import AdaptiveAugmentationBuilder

def load_config(config_path='config/config.yaml'):
    """Loads the configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    writer = SummaryWriter('runs/alphabet_experiment')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    print(f"{'='*60}")

    aug_builder = AdaptiveAugmentationBuilder(base_size=config['data']['image_size'])
    
    train_transform = aug_builder.build_train_transform(
        (config['data']['image_size'], config['data']['image_size'])
    )
    # train_transform = aug_builder.build_val_transform(
    #     (config['data']['image_size'], config['data']['image_size'])
    # )
    val_transform = aug_builder.build_val_transform(
        (config['data']['image_size'], config['data']['image_size'])
    )

    print("📂 Loading dataset...")



    # full_dataset = datasets.ImageFolder(
    #     root=config['data']['train_root'], 
    #     transform=train_transform
    # )
    
    # class_names = full_dataset.classes
    # num_classes = len(class_names)
    # print(f"📚 Found classes: {num_classes}")
    # print(f"   {class_names}")
    
    # # Count number of images per class
    # class_counts = {}
    # for idx, (_, label) in enumerate(full_dataset.samples):
    #     class_name = class_names[label]
    #     class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # print(f"\n📊 Class distribution:")
    # for name, count in sorted(class_counts.items()):
    #     print(f"   {name}: {count} images")
    

    # train_size = int(0.8 * len(full_dataset))
    # val_size = len(full_dataset) - train_size
    # train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # print(f"Train size: {len(train_dataset)}")
    # print(f"Val size: {len(val_dataset)}")

    # log_transformed_images(writer, train_dataset, num_samples=128, tag="train_augmented")

    # # Changing the transformation for validation
    # val_dataset.dataset.transform = val_transform
    
    # print(f"\n Train samples: {len(train_dataset)}")
    # print(f"Val samples: {len(val_dataset)}")
    
    # train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['data']['num_workers'], pin_memory=True)
    # val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'], pin_memory=True)


    train_dataset = datasets.ImageFolder(
        root=config['data']['train_root'], 
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=config['data']['val_root'],
        transform=val_transform
    )
    
    class_names = train_dataset.classes  # Берем классы из train
    num_classes = len(class_names)
    print(f"📚 Found classes: {num_classes}")
    print(f"   {class_names}")
    
    # Проверка, что классы совпадают в train и val
    if set(train_dataset.classes) != set(val_dataset.classes):
        print("⚠️ WARNING: Train and Val datasets have different classes!")
        print(f"Train classes: {set(train_dataset.classes)}")
        print(f"Val classes: {set(val_dataset.classes)}")
        
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Логируем распределение классов
    print(f"\n📊 Train class distribution:")
    for class_name in class_names:
        count = sum(1 for _, label in train_dataset.samples if train_dataset.classes[label] == class_name)
        print(f"   {class_name}: {count} images")
    
    print(f"\n📊 Val class distribution:")
    for class_name in class_names:
        count = sum(1 for _, label in val_dataset.samples if val_dataset.classes[label] == class_name)
        print(f"   {class_name}: {count} images")
    
    # log_transformed_images(writer, train_dataset, num_samples=128, tag="train_augmented")
    # log_transformed_images(writer, val_dataset, num_samples=64, tag="val_original")

    import random

    fig, axes = plt.subplots(16, 16, figsize=(16, 16))

    # Выбираем 256 случайных индексов
    num_samples = 256
    dataset_size = len(train_dataset)
    random_indices = random.sample(range(dataset_size), num_samples)

    for i, idx in enumerate(random_indices):
        img, label = train_dataset[idx]
        row = i // 16
        col = i % 16
        axes[row, col].imshow(img.squeeze(), cmap='gray')
        axes[row, col].set_title(f'{class_names[label]}', fontsize=6)
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()

    print(f"Image shape: {img.shape}") 


    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=config['data']['num_workers'], 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        num_workers=config['data']['num_workers'], 
        pin_memory=True
    )

    sample, _ = next(iter(train_loader))
    print(f"Input range: min={sample.min():.2f}, max={sample.max():.2f}, dtype={sample.dtype}")
    print(f"Unique values: {torch.unique(sample)}")


    log_transformed_images_from_dataloader(writer, train_loader, num_samples=512, tag="train_augmented")

    log_transformed_images_from_dataloader(writer, val_loader, num_samples=512, tag="val_original")


    # Выбираем модель
    use_pretrained = True
    model_name = 'efficientnet_b0'  # или 'efficientnet_b0', 'mobilenet_v3_small', 'resnet34' resnet18
    
    trainer = None

    if use_pretrained:
        from models.pretrained_model import AlphabetRecognizerPretrained
        
        model = AlphabetRecognizerPretrained(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=True
        )

        model.to(device)
        
        # Вариант 1: Разные learning rates
        trainer = ModelTrainer(model, device, config, writer)
        
        # Устанавливаем разные LR для backbone и классификатора
        backbone_lr = config['training'].get('backbone_lr', 1e-5)
        classifier_lr = config['training']['learning_rate']

        print(backbone_lr, classifier_lr)

        trainer.set_different_lrs(backbone_lr, classifier_lr)
        
    else:
        from models.model import AlphabetRecognizer
        model = AlphabetRecognizer(num_classes=num_classes)
    
        model.to(device)
    
        # Calculating parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n🧠 Model: AlphabetRecognizer")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")

        trainer = ModelTrainer(model, device, config, writer)


    early_stopping = EarlyStopping(patience=config['training']['early_stopping_patience'])
    
    trainer.train(train_loader, val_loader, class_names, early_stopping)

    # if hasattr(model, 'unfreeze_stage'):
    #     trainer.train_with_gradual_unfreezing(train_loader, val_loader, class_names, early_stopping)
    # else:
    #     trainer.train(train_loader, val_loader, class_names, early_stopping)

if __name__ == "__main__":
    main()