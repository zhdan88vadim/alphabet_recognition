from data.augmentation import ExtractLetterWithMargin, SquarePad
from training.callbacks import EarlyStopping
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from models.emnist_model import AlphabetRecognizerEmnist
from training.trainer import ModelTrainer
import yaml
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


def load_config(config_path='config/config.yaml'):
    """Loads the configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()    
    writer = SummaryWriter('runs/alphabet_experiment')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Трансформации
    transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),
        # ExtractLetterWithMargin(margin=4, fill_white=True),
        # CenterDigitsTransform(padding=10, fill_value=255),
        # SquarePad(fill_white=True),
        # transforms.Resize((28, 28)),

        # transforms.RandomRotation(180),  # Поворот на 90°
        # transforms.RandomHorizontalFlip(p=1.0),  # Зеркальное отражение (всегда)
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # EMNIST
    emnist = datasets.EMNIST(
        root='../emnist_data',
        split='letters',
        train=True,
        download=True,
        transform=transform
    )
    
    # Разделение train/val
    train_size = int(0.8 * len(emnist))
    val_size = len(emnist) - train_size
    train_dataset, val_dataset = random_split(emnist, [train_size, val_size])
        

        # Показать примеры
    fig, axes = plt.subplots(16, 16, figsize=(16, 16))
    for i in range(254):
        img, label = train_dataset[i]
        row = i // 16
        col = i % 16
        axes[row, col].imshow(img.squeeze(), cmap='gray')
        axes[row, col].set_title(f'{emnist.classes[label]}', fontsize=6)
        axes[row, col].axis('off')

    for i in range(254, 256):
        row = i // 16
        col = i % 16
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()

    # Проверьте размеры
    print(f"Image shape: {img.shape}") 

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Модель
    model = AlphabetRecognizerEmnist(num_classes=26)  # A-Z
    model.to(device)
    
    # Тренировка
    trainer = ModelTrainer(model, device, config=config, writer=writer)
    early_stopping = EarlyStopping(patience=config['training']['early_stopping_patience'])
    
    trainer.train(train_loader, val_loader, class_names=[chr(ord('A')+i) for i in range(26)], early_stopping=early_stopping)


if __name__ == "__main__":
    main()