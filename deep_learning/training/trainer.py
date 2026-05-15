import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import copy
import json
import numpy as np
import mlflow
import mlflow.pytorch

class ModelTrainer:
    """Класс для обучения моделей с MLflow логированием"""
    
    def __init__(self, model, device, config, writer, use_mlflow=True):
        self.model = model
        self.device = device
        self.config = config
        self.writer = writer
        self.use_mlflow = use_mlflow
        
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
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.best_acc = 0.0
        self.best_model_wts = copy.deepcopy(model.state_dict())
        
        # Инициализация MLflow
        if self.use_mlflow:
            self._setup_mlflow()

    def _setup_mlflow(self):
        """Настройка MLflow эксперимента"""
        if not self.use_mlflow:
            return
        
        # Устанавливаем tracking URI - используем SQLite вместо файловой системы
        # (убираем предупреждение о депрекации)
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        
        # Создаём или получаем эксперимент
        experiment_name = self.config.get('mlflow_experiment_name', 'alphabet_recognition')
        
        # Проверяем существует ли эксперимент
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
        
        mlflow.set_experiment(experiment_name)
        
        print(f"🔧 MLflow эксперимент: {experiment_name}")
        print(f"   Tracking URI: sqlite:///mlflow.db")
        
    
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
    
    # TODO: move to metrics file
    
    def log_confusion_matrix(self, val_loader, class_names, epoch):
        """Логирует confusion matrix в MLflow"""
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Строим confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Визуализируем
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix (Epoch {epoch})')
        
        # Логируем в MLflow
        if self.use_mlflow:
            mlflow.log_figure(fig, f"confusion_matrix_epoch_{epoch}.png")
        
        plt.close()
    
    def train(self, train_loader, val_loader, class_names, early_stopping):
        """Основной цикл обучения с MLflow логированием"""
        epochs = self.config['training']['epochs']
        
        print(f"\n Начинаем обучение на {epochs} эпох...")
        
        # Проверяем, активен ли уже run
        active_run = mlflow.active_run()
        
        if self.use_mlflow and not active_run:
            # Запускаем новый run только если нет активного
            run_name = self.config.get('run_name', f"run_{self.best_acc:.2f}")
            mlflow.start_run(run_name=run_name)
            print(f"🚀 Started MLflow run: {run_name}")
        elif self.use_mlflow and active_run:
            print(f"⚠️ Using existing MLflow run: {active_run.info.run_id}")
        
        try:
            for epoch in range(epochs):
                train_loss, train_acc = self.train_epoch(train_loader)
                val_loss, val_acc = self.validate(val_loader)
                
                # Сохраняем историю
                self.train_losses.append(train_loss)
                self.train_accs.append(train_acc)
                self.val_losses.append(val_loss)
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
                
                # MLflow логи
                if self.use_mlflow:
                    mlflow.log_metric("train_loss", train_loss, step=epoch)
                    mlflow.log_metric("train_accuracy", train_acc, step=epoch)
                    mlflow.log_metric("val_loss", val_loss, step=epoch)
                    mlflow.log_metric("val_accuracy", val_acc, step=epoch)
                    
                    # Логируем learning rate
                    current_lr = self.optimizer.param_groups[0]['lr']
                    mlflow.log_metric("learning_rate", current_lr, step=epoch)
                
                # Каждые N эпох логируем confusion matrix
                if (epoch + 1) % 5 == 0 and self.use_mlflow:
                    self.log_confusion_matrix(val_loader, class_names, epoch + 1)
                
                # Scheduler и Early Stopping
                self.scheduler.step(val_acc)
                
                # Сохраняем лучшую модель
                if val_acc > self.best_acc:
                    self.best_acc = val_acc
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())
                    self._save_checkpoint(epoch, val_acc, class_names)
                    print(f"  ✨ Сохранена лучшая модель (Acc: {val_acc:.2f}%)")
                

                    # Логируем лучшую модель в MLflow
                    if self.use_mlflow:
                        mlflow.log_metric("best_val_accuracy", val_acc, step=epoch)
                        
                        # Создаем пример входных данных для трассировки
                        # Берем один батч из train_loader
                        sample_input, _ = next(iter(train_loader))
                        sample_input = sample_input[:1].to(self.device)  # Берем один пример
                        
                        model_name = f"alphabet_model_{self.best_acc:.2f}".replace('.', '_')
                        
                        # Сохраняем модель с input_example
                        mlflow.pytorch.log_model(
                            pytorch_model=self.model,
                            name="best_model",
                            registered_model_name=model_name,
                            serialization_format='pt2',  # безопасный формат
                            input_example=sample_input,  # ← обязательно для pt2
                            pip_requirements=["torch>=2.0.0", "torchvision>=0.15.0"]
                        )
                
                # Early stopping
                if early_stopping(val_acc):
                    print(f"\n🛑 Early stopping на эпохе {epoch+1}")
                    break
                    
                # Дополнительная проверка: если точность валидации падает 3 эпохи подряд
                if len(self.val_accs) > 3:
                    if self.val_accs[-1] < self.val_accs[-2] < self.val_accs[-3]:
                        print(f"  ⚠️ Val accuracy падает 3 эпохи подряд!")
                        
        finally:
            # Завершаем MLflow run только если мы его начинали
            if self.use_mlflow and mlflow.active_run():
                # Логируем финальные метрики
                mlflow.log_metric("final_train_accuracy", self.train_accs[-1] if self.train_accs else 0)
                mlflow.log_metric("final_val_accuracy", self.val_accs[-1] if self.val_accs else 0)
                mlflow.log_metric("best_val_accuracy", self.best_acc)
                
                # Логируем графики
                self._log_training_plots()
                
                mlflow.end_run()
                print("✅ MLflow run завершён")
        
        # Загружаем лучшую модель
        self.model.load_state_dict(self.best_model_wts)
        self._save_history()
        
        print(f"\n✅ Обучение завершено!")
        print(f"🏆 Лучшая точность: {self.best_acc:.2f}%")
        
        return self.model
    
    def _log_training_plots(self):
        """Логирует графики обучения в MLflow"""
        import matplotlib.pyplot as plt
        
        # График loss
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        mlflow.log_figure(fig1, "training_validation_loss.png")
        
        # График accuracy
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(self.train_accs, label='Train Accuracy')
        ax2.plot(self.val_accs, label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        mlflow.log_figure(fig2, "training_validation_accuracy.png")
        
        plt.close('all')
    
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
                'train_accs': self.train_accs,
                'val_losses': self.val_losses,
                'val_accs': self.val_accs,
                'best_acc': self.best_acc
            }, f)