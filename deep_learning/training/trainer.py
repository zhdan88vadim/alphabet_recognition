import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import copy
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.pytorch
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

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
        
        # Дополнительные метрики
        self.train_f1_scores = []
        self.val_f1_scores = []
        self.train_precisions = []
        self.val_precisions = []
        self.train_recalls = []
        self.val_recalls = []
        
        self.best_acc = 0.0
        self.best_f1 = 0.0
        self.best_model_wts = copy.deepcopy(model.state_dict())
        
        # Инициализация MLflow
        if self.use_mlflow:
            self._setup_mlflow()

    def _setup_mlflow(self):
        """Настройка MLflow эксперимента"""
        if not self.use_mlflow:
            return
        
        # Устанавливаем tracking URI - используем SQLite вместо файловой системы
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
        
        # Логируем параметры модели
        mlflow.log_params({
            "model_architecture": self.model.__class__.__name__,
            "learning_rate": self.config['training']['learning_rate'],
            "weight_decay": self.config['training']['weight_decay'],
            "batch_size": self.config['data'].get('batch_size', 32),
            "epochs": self.config['training']['epochs'],
            "optimizer": "AdamW",
            "scheduler": "ReduceLROnPlateau"
        })
        
        print(f"🔧 MLflow эксперимент: {experiment_name}")
        print(f"   Tracking URI: sqlite:///mlflow.db")
    
    def calculate_metrics(self, outputs, labels, average='macro'):
        """Рассчитывает дополнительные метрики"""
        _, predicted = torch.max(outputs.data, 1)
        predicted_np = predicted.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        # F1 Score
        f1 = f1_score(labels_np, predicted_np, average=average, zero_division=0)
        
        # Precision и Recall
        precision = precision_score(labels_np, predicted_np, average=average, zero_division=0)
        recall = recall_score(labels_np, predicted_np, average=average, zero_division=0)
        
        return f1, precision, recall
    
    def train_epoch(self, train_loader):
        """Обучает одну эпоху"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Для расчета метрик за эпоху
        all_outputs = []
        all_labels = []
        
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
            
            # Сохраняем для расчета метрик
            all_outputs.append(outputs.detach().cpu())
            all_labels.append(labels.detach().cpu())
            
            loop.set_postfix(loss=loss.item(), acc=f"{100*correct/total:.1f}%")
        
        # Объединяем все предсказания для расчета метрик
        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Рассчитываем метрики
        f1, precision, recall = self.calculate_metrics(all_outputs, all_labels)
        
        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy, f1, precision, recall
    
    def validate(self, val_loader):
        """Валидация с дополнительными метриками"""
        self.model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Сохраняем для расчета метрик
                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())
        
        # Объединяем все предсказания
        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Рассчитываем метрики
        f1, precision, recall = self.calculate_metrics(all_outputs, all_labels)
        
        avg_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy, f1, precision, recall
    
    def log_confusion_matrix(self, val_loader, class_names, epoch):
        """Логирует confusion matrix в MLflow"""
        from sklearn.metrics import confusion_matrix
        
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
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix (Epoch {epoch})')
        
        # Логируем в MLflow
        if self.use_mlflow:
            mlflow.log_figure(fig, f"confusion_matrix_final.png")
        
        plt.savefig(f"../readme_images/confusion_matrix_final.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Логируем classification report
        # if self.use_mlflow and epoch % 5 == 0:  # Логируем каждые 5 эпох
        #     report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
        #     with open(f"classification_report_epoch_{epoch}.txt", "w") as f:
        #         f.write(report)
        #     mlflow.log_artifact(f"classification_report_epoch_{epoch}.txt")
    
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
                train_loss, train_acc, train_f1, train_precision, train_recall = self.train_epoch(train_loader)
                val_loss, val_acc, val_f1, val_precision, val_recall = self.validate(val_loader)
                
                # Сохраняем историю
                self.train_losses.append(train_loss)
                self.train_accs.append(train_acc)
                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)
                
                self.train_f1_scores.append(train_f1)
                self.val_f1_scores.append(val_f1)
                self.train_precisions.append(train_precision)
                self.val_precisions.append(val_precision)
                self.train_recalls.append(train_recall)
                self.val_recalls.append(val_recall)
                
                # Отчет
                print(f"\nEpoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | F1: {train_f1:.4f}")
                print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | F1: {val_f1:.4f}")
                print(f"  Precision: {val_precision:.4f} | Recall: {val_recall:.4f}")
                
                # TensorBoard логи
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_acc, epoch)
                self.writer.add_scalar('Accuracy/val', val_acc, epoch)
                self.writer.add_scalar('F1/train', train_f1, epoch)
                self.writer.add_scalar('F1/val', val_f1, epoch)
                self.writer.add_scalar('Precision/val', val_precision, epoch)
                self.writer.add_scalar('Recall/val', val_recall, epoch)
                
                # MLflow логи
                if self.use_mlflow:
                    mlflow.log_metric("train_loss", train_loss, step=epoch)
                    mlflow.log_metric("train_accuracy", train_acc, step=epoch)
                    mlflow.log_metric("train_f1_score", train_f1, step=epoch)
                    mlflow.log_metric("train_precision", train_precision, step=epoch)
                    mlflow.log_metric("train_recall", train_recall, step=epoch)
                    
                    mlflow.log_metric("val_loss", val_loss, step=epoch)
                    mlflow.log_metric("val_accuracy", val_acc, step=epoch)
                    mlflow.log_metric("val_f1_score", val_f1, step=epoch)
                    mlflow.log_metric("val_precision", val_precision, step=epoch)
                    mlflow.log_metric("val_recall", val_recall, step=epoch)
                    
                    # Логируем learning rate
                    current_lr = self.optimizer.param_groups[0]['lr']
                    mlflow.log_metric("learning_rate", current_lr, step=epoch)
                
                # Scheduler и Early Stopping
                self.scheduler.step(val_acc)
                
                # Сохраняем лучшую модель (по F1 или Accuracy)
                if val_f1 > self.best_f1:
                    self.best_f1 = val_f1
                    self.best_acc = val_acc
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())
                    self._save_checkpoint(epoch, val_acc, val_f1, class_names)
                    print(f"  ✨ Сохранена лучшая модель (Acc: {val_acc:.2f}%, F1: {val_f1:.4f})")
                
                    # Логируем лучшие метрики в MLflow
                    if self.use_mlflow:
                        mlflow.log_metric("best_val_accuracy", val_acc, step=epoch)
                        mlflow.log_metric("best_val_f1_score", val_f1, step=epoch)
                        mlflow.log_metric("best_val_precision", val_precision, step=epoch)
                        mlflow.log_metric("best_val_recall", val_recall, step=epoch)
                        
                        # Логируем confusion matrix для лучшей модели
                        # if epoch > 5:  # Начиная с 5 эпохи
                        #     self.log_confusion_matrix(val_loader, class_names, epoch + 1)
                        
                        # Сохраняем модель в MLflow
                        model_name = f"alphabet_model_f1_{self.best_f1:.4f}_acc_{self.best_acc:.2f}".replace('.', '_')
                        
                        # Сохраняем модель
                        mlflow.pytorch.log_model(
                            pytorch_model=self.model,
                            artifact_path="best_model",
                            registered_model_name=model_name,
                            serialization_format='pickle',
                            pip_requirements=["torch>=2.0.0", "torchvision>=0.15.0", "scikit-learn>=1.0.0"]
                        )
                
                # Early stopping (теперь на основе F1 вместо accuracy)
                if early_stopping(val_f1):
                    print(f"\n🛑 Early stopping на эпохе {epoch+1}")
                    break
                    
                # Дополнительная проверка: если F1 падает 3 эпохи подряд
                if len(self.val_f1_scores) > 3:
                    if self.val_f1_scores[-1] < self.val_f1_scores[-2] < self.val_f1_scores[-3]:
                        print(f"  ⚠️ Val F1 падает 3 эпохи подряд!")
                        
        finally:
            # Завершаем MLflow run только если мы его начинали
            if self.use_mlflow and mlflow.active_run():
                # Логируем финальные метрики
                if self.train_accs:
                    mlflow.log_metric("final_train_accuracy", self.train_accs[-1])
                    mlflow.log_metric("final_val_accuracy", self.val_accs[-1])
                    mlflow.log_metric("final_train_f1", self.train_f1_scores[-1])
                    mlflow.log_metric("final_val_f1", self.val_f1_scores[-1])
                
                mlflow.log_metric("best_val_accuracy", self.best_acc)
                mlflow.log_metric("best_val_f1_score", self.best_f1)
                
                # Логируем графики
                self._log_training_plots()
                
                # Логируем финальный classification report
                self._log_final_classification_report(val_loader, class_names)

                self.log_confusion_matrix(val_loader, class_names, epoch + 1)
                
                mlflow.end_run()
                print("✅ MLflow run завершён")
        
        # Загружаем лучшую модель
        self.model.load_state_dict(self.best_model_wts)
        self._save_history()
        
        print(f"\n✅ Обучение завершено!")
        print(f"🏆 Лучшая точность: {self.best_acc:.2f}%")
        print(f"🏆 Лучший F1 Score: {self.best_f1:.4f}")
        
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
        plt.savefig('../readme_images/training_validation_loss.png', dpi=150, bbox_inches='tight')
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
        plt.savefig('../readme_images/training_validation_accuracy.png', dpi=150, bbox_inches='tight')
        mlflow.log_figure(fig2, "training_validation_accuracy.png")
        
        # График F1 Score
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.plot(self.train_f1_scores, label='Train F1 Score')
        ax3.plot(self.val_f1_scores, label='Val F1 Score')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score')
        ax3.set_title('Training and Validation F1 Score')
        ax3.legend()
        ax3.grid(True)
        plt.savefig('../readme_images/training_validation_f1.png', dpi=150, bbox_inches='tight')
        mlflow.log_figure(fig3, "training_validation_f1.png")
        
        # График Precision/Recall
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        ax4.plot(self.val_precisions, label='Val Precision')
        ax4.plot(self.val_recalls, label='Val Recall')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Score')
        ax4.set_title('Validation Precision and Recall')
        ax4.legend()
        ax4.grid(True)
        plt.savefig('../readme_images/validation_precision_recall.png', dpi=150, bbox_inches='tight')
        mlflow.log_figure(fig4, "validation_precision_recall.png")
        
        plt.close('all')
    
    def _log_final_classification_report(self, val_loader, class_names):
        """Логирует финальный classification report в MLflow"""
        from sklearn.metrics import classification_report
        
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
        
        # Создаем classification report
        report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
        
        # Сохраняем как текст
        with open("final_classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("final_classification_report.txt")
        
        # Логируем как метрики для каждого класса
        report_dict = classification_report(all_labels, all_preds, target_names=class_names, 
                                           output_dict=True, zero_division=0)

        data = []
        for class_name, metrics in report_dict.items():
            if isinstance(metrics, dict) and class_name not in ['macro avg', 'weighted avg', 'accuracy']:
                data.append({
                    'Class': class_name,
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1-score']
                })
        
        df = pd.DataFrame(data)
        
        # График: Precision и Recall для каждой буквы
        fig, ax = plt.subplots(figsize=(20, 6))
        x = np.arange(len(df))
        width = 0.35
        
        ax.bar(x - width/2, df['Precision'], width, label='Precision', color='lightblue')
        ax.bar(x + width/2, df['Recall'], width, label='Recall', color='lightcoral')
        
        ax.set_xlabel('Letter')
        ax.set_ylabel('Score')
        ax.set_title('Precision and Recall per Letter')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Class'], rotation=45)
        ax.legend()
        ax.axhline(y=0.9, color='green', linestyle='--', label='Good Threshold (0.9)')
        ax.axhline(y=0.7, color='orange', linestyle='--', label='Warning Threshold (0.7)')
        
        plt.tight_layout()
        plt.savefig('../readme_images/metrics/val/per_letter_metrics.png')
        mlflow.log_figure(fig, 'per_letter_metrics.png')
        plt.close()
        
        # Сохраняем CSV с метриками для Excel/Tableau
        # df.to_csv('per_letter_metrics.csv', index=False)
        # mlflow.log_artifact('per_letter_metrics.csv')
    
    def _save_checkpoint(self, epoch, val_acc, val_f1, class_names):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'val_f1': val_f1,
            'class_names': class_names,
        }, 'best_alphabet_model.pth')
    
    def _save_history(self):
        with open("training_history.json", "w", encoding="utf-8") as f:
            json.dump({
                'train_losses': self.train_losses,
                'train_accs': self.train_accs,
                'val_losses': self.val_losses,
                'val_accs': self.val_accs,
                'train_f1_scores': self.train_f1_scores,
                'val_f1_scores': self.val_f1_scores,
                'train_precisions': self.train_precisions,
                'val_precisions': self.val_precisions,
                'train_recalls': self.train_recalls,
                'val_recalls': self.val_recalls,
                'best_acc': self.best_acc,
                'best_f1': self.best_f1
            }, f)