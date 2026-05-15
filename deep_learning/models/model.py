import torch
import torch.nn as nn

class AlphabetRecognizer(nn.Module):
    def __init__(self, num_classes=33):
        super(AlphabetRecognizer, self).__init__()
        
        # Conv layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)

        # 64 -> 32 -> 16 -> 8
        self.fc1 = nn.Linear(128 * 8 * 8, 512)  # 128*8*8 = 8,192
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
        # 64x64x3 -> 32x32x32
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        # 32x32x32 -> 16x16x64
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        # 16x16x64 -> 8x8x128
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        # Flatten: 8x8x128 = 8,192
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x