from models.emnist_model import AlphabetRecognizerEmnist
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загрузка модели
model = AlphabetRecognizerEmnist(num_classes=26)
checkpoint = torch.load('best_alphabet_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(device)

# Загрузка EMNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.EMNIST(
    root='../emnist_data',
    split='letters',
    train=False,
    download=False,
    transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ✅ Полная проверка на всех данных
def evaluate_full(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels - 1  # Сдвиг 1-26 → 0-25
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    print(f"✅ Test Accuracy: {accuracy:.2f}%")
    print(f"   Total samples: {total}")
    print(f"   Correct: {correct}")
    print(f"   Incorrect: {total - correct}")
    
    return accuracy, all_preds, all_labels

# Запуск полной проверки
accuracy, preds, labels = evaluate_full(model, test_loader, device)

# 📊 Classification Report
class_names = [chr(ord('A') + i) for i in range(26)]
print("\n📊 Classification Report:")
print(classification_report(labels, preds, target_names=class_names, zero_division=0))

# 🔍 Confusion Matrix
cm = confusion_matrix(labels, preds)
plt.figure(figsize=(20, 20))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - EMNIST Letters')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix_emnist.png', dpi=150, bbox_inches='tight')
plt.show()

# 📈 Per-class accuracy
print("\n📈 Per-class accuracy:")
class_correct = np.zeros(26)
class_total = np.zeros(26)
for i in range(len(labels)):
    class_total[labels[i]] += 1
    if preds[i] == labels[i]:
        class_correct[labels[i]] += 1

for i in range(26):
    if class_total[i] > 0:
        acc = 100 * class_correct[i] / class_total[i]
        print(f"  {chr(ord('A') + i)}: {acc:.2f}% ({int(class_correct[i])}/{int(class_total[i])})")
    else:
        print(f"  {chr(ord('A') + i)}: No samples")