import torch
import torch.nn as nn

class AlphabetRecognizerEmnist(nn.Module):
    def __init__(self, num_classes=26):  # EMNIST letters: A-Z = 26 классов
        super(AlphabetRecognizerEmnist, self).__init__()
        
        # Conv layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)

        # ✅ Для EMNIST: 28 -> 14 -> 7 -> 3 (после 3 сверток)
        # 128 * 3 * 3 = 1152
        self.fc1 = nn.Linear(128 * 3 * 3, 512)  # 1152 → 512
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 28x28 -> 14x14x32
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        # 14x14x32 -> 7x7x64
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        # 7x7x64 -> 3x3x128
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        # Flatten: 3x3x128 = 1152
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x