import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import copy
import json
from torch.utils.tensorboard import SummaryWriter
import numpy as np

class ModelTrainer:
    """Класс для обучения моделей"""
    
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5,
            patience=config['training']['scheduler_patience']
        )
        
        # History
        self.train_losses = []
        self.val_accs = []
        self.best_acc = 0.0
        self.best_model_wts = copy.deepcopy(model.state_dict())
        
        # TensorBoard
        self.writer = SummaryWriter('runs/alphabet_experiment')
    
    def train_epoch(self, train_loader):
        """Обучает одну эпоху"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc="Training")
        for inputs, labels in loop:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['training']['gradient_clip_norm']
            )
            
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=f"{100*correct/total:.1f}%")
        
        return running_loss / len(train_loader), 100 * correct / total
    
    def validate(self, val_loader):
        """Валидация"""
        self.model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return val_loss / len(val_loader), 100 * correct / total
    
    def train(self, train_loader, val_loader, class_names, early_stopping):
        """Основной цикл обучения"""
        epochs = self.config['training']['epochs']
        
        print(f"\n Начинаем обучение на {epochs} эпох...")
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            # Сохраняем историю
            self.train_losses.append(train_loss)
            self.val_accs.append(val_acc)
            
            # Отчет
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # TensorBoard логи
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            
            # Scheduler и Early Stopping
            self.scheduler.step(val_acc)
            
            # Сохраняем лучшую модель
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                self._save_checkpoint(epoch, val_acc, class_names)
                print(f"  ✨ Сохранена лучшая модель (Acc: {val_acc:.2f}%)")
            
            # Early stopping
            if early_stopping(val_acc):
                print(f"\n🛑 Early stopping на эпохе {epoch+1}")
                break
                
            # Дополнительная проверка: если точность валидации падает 3 эпохи подряд
            if len(self.val_accs) > 3:
                if self.val_accs[-1] < self.val_accs[-2] < self.val_accs[-3]:
                    print(f"  ⚠️ Val accuracy падает 3 эпохи подряд!")            
            
        # Загружаем лучшую модель
        self.model.load_state_dict(self.best_model_wts)
        self._save_history()
        
        print(f"\n✅ Обучение завершено!")
        print(f"🏆 Лучшая точность: {self.best_acc:.2f}%")
        
        return self.model
    
    def _save_checkpoint(self, epoch, val_acc, class_names):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'class_names': class_names,
        }, 'best_alphabet_model.pth')
    
    def _save_history(self):
        with open("training_history.json", "w", encoding="utf-8") as f:
            json.dump({
                'train_losses': self.train_losses,
                'val_accs': self.val_accs,
                'best_acc': self.best_acc
            }, f)