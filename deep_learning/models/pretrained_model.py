import torch
import torch.nn as nn
import torchvision.models as models

class AlphabetRecognizerPretrained(nn.Module):
    def __init__(self, num_classes=33, model_name='efficientnet_b0', pretrained=True, image_size=224):
        super(AlphabetRecognizerPretrained, self).__init__()
        
        # Загружаем предобученную модель
        if model_name == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            # ✅ ВАЖНО: Заменяем первый слой для 1 канала (grayscale)
            original_conv = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(
                in_channels=1,  # Grayscale
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            # Инициализируем веса (можно копировать среднее по 3 каналам)
            with torch.no_grad():
                self.backbone.features[0][0].weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
            
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        elif model_name == 'mobilenet_v3_small':
            self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
            # Аналогично для MobileNet
            original_conv = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(1, original_conv.out_channels, 3, 2, 1, bias=False)
            with torch.no_grad():
                self.backbone.features[0][0].weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
            
            in_features = self.backbone.classifier[3].in_features
            self.backbone.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # ✅ Улучшенный классификатор для букв
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
        
        self._freeze_backbone(freeze=True)
        
    def _freeze_backbone(self, freeze=True):
        """Замораживаем backbone для transfer learning"""
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("🔒 Backbone frozen (feature extractor mode)")
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("🔓 Backbone unfrozen (fine-tuning mode)")
    
    def forward(self, x):
        # Если входные данные 1 канал (grayscale) - уже обработано в первом слое
        # Не нужно repeat(1, 3, 1, 1)!
        
        features = self.backbone(x)
        output = self.classifier(features)
        return output