import cv2
from data.augmentation import ExtractLetterWithMargin, Invert, SimpleThinOrThicken
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import json
import os
from datetime import datetime
import pandas as pd
from pathlib import Path

class DatasetTester:
    def __init__(self, model_path='best_alphabet_model.pth', mapping_path='class_mapping.json'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        self.model, self.class_names = self._load_model(model_path, mapping_path)
        
        self.transform = transforms.Compose([
            ExtractLetterWithMargin(margin=4, fill_white=True),
            transforms.Resize((64, 64)),
            Invert(),
            SimpleThinOrThicken(p=1, strength='light', min_thickness=1),
            Invert(),
            transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Создаем папку для результатов
        self.results_dir = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def _load_model(self, model_path, mapping_path):
        # with open(mapping_path, 'r', encoding='utf-8') as f:
        #     class_names = json.load(f)

        checkpoint = torch.load(model_path, map_location=self.device)        
        class_names = checkpoint['class_names']
        
        print(f"Count of classes: {len(class_names)}")

        from models.model import AlphabetRecognizer
        model = AlphabetRecognizer()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"Model loaded: {model_path}")
        return model, class_names
    
    def preprocess_letter(self, letter_image):
        """Предобработка отдельной буквы"""
        # Бинаризация
        _, binary = cv2.threshold(letter_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Инвертируем обратно для модели
        binary = cv2.bitwise_not(binary)
        
        # Преобразуем в PIL
        img_pil = Image.fromarray(binary)
        
        # Применяем трансформации
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        
        return img_tensor
    
    def predict_letter(self, letter_image):
        """Предсказание одной буквы"""
        img_tensor = self.preprocess_letter(letter_image)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            
            # Получаем топ-3 предсказания
            top3_prob, top3_idx = torch.topk(probs, min(3, len(self.class_names)))
            top3 = [(self.class_names[idx.item()], prob.item() * 100) 
                    for idx, prob in zip(top3_idx[0], top3_prob[0])]
        
        predicted_letter = self.class_names[predicted.item()]
        confidence_score = confidence.item() * 100
        
        return predicted_letter, confidence_score, top3
    
    def test_dataset(self, dataset_path):
        """Тестирование на всем датасете"""
        dataset_path = Path(dataset_path)
        results = []
        
        # Получаем все папки с буквами
        letter_folders = sorted([f for f in dataset_path.iterdir() if f.is_dir()])
        
        print(f"\n{'='*70}")
        print(f"Тестирование датасета: {dataset_path}")
        print(f"Найдено букв: {len(letter_folders)}")
        print(f"{'='*70}\n")
        
        for letter_folder in letter_folders:
            true_letter = letter_folder.name
            print(f"\n📁 Тестируем букву: {true_letter}")
            
            # Получаем все изображения этой буквы
            image_files = list(letter_folder.glob("*.png")) + \
                         list(letter_folder.glob("*.jpg")) + \
                         list(letter_folder.glob("*.jpeg")) + \
                         list(letter_folder.glob("*.bmp"))
            
            print(f"   Найдено изображений: {len(image_files)}")
            
            for img_path in image_files:
                # Загружаем изображение
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"   ❌ Не удалось загрузить: {img_path.name}")
                    continue
                
                # Конвертируем в grayscale
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image.copy()
                
                # Предсказываем
                predicted_letter, confidence, top3 = self.predict_letter(gray)
                
                # Сохраняем результат
                results.append({
                    'true_letter': true_letter,
                    'predicted_letter': predicted_letter,
                    'confidence': confidence,
                    'is_correct': true_letter == predicted_letter,
                    'image_name': img_path.name,
                    'image_path': str(img_path),
                    'top3': top3
                })
                
                status = "✅" if true_letter == predicted_letter else "❌"
                print(f"   {status} {img_path.name}: {predicted_letter} ({confidence:.1f}%)")
        
        return results
    
    def create_results_table(self, results):
        """Создание таблицы с результатами"""
        df = pd.DataFrame(results)
        
        # Группировка по буквам для статистики
        stats = df.groupby('true_letter').agg({
            'is_correct': ['count', 'sum', 'mean'],
            'confidence': 'mean'
        }).round(4)
        
        stats.columns = ['total', 'correct', 'accuracy', 'avg_confidence']
        stats['accuracy'] = stats['accuracy'] * 100
        stats['error_rate'] = 100 - stats['accuracy']
        
        # Общая статистика
        total_accuracy = df['is_correct'].mean() * 100
        avg_confidence = df['confidence'].mean()
        
        # Сохраняем в CSV
        df.to_csv(f"{self.results_dir}/detailed_results.csv", index=False, encoding='utf-8-sig')
        stats.to_csv(f"{self.results_dir}/summary_stats.csv", encoding='utf-8-sig')
        
        return df, stats, total_accuracy, avg_confidence
    
    def visualize_results_grid(self, results, max_per_letter=3):
        """Создание сетки с результатами"""
        # Группируем по истинным буквам
        results_by_letter = {}
        for r in results:
            if r['true_letter'] not in results_by_letter:
                results_by_letter[r['true_letter']] = []
            results_by_letter[r['true_letter']].append(r)
        
        # Сортируем буквы
        letters = sorted(results_by_letter.keys())
        
        # Определяем размеры сетки
        n_letters = len(letters)
        n_cols = min(6, n_letters)
        n_rows = (n_letters + n_cols - 1) // n_cols
        
        # Создаем фигуру
        fig = plt.figure(figsize=(n_cols * 4, n_rows * 6))
        
        for idx, letter in enumerate(letters):
            # Берем первые max_per_letter изображений
            letter_results = results_by_letter[letter][:max_per_letter]
            
            # Создаем подграфик для буквы
            ax = plt.subplot(n_rows, n_cols, idx + 1)
            
            # Загружаем и отображаем изображения
            n_images = len(letter_results)
            for i, result in enumerate(letter_results):
                # Загружаем изображение
                img = cv2.imread(result['image_path'], cv2.IMREAD_GRAYSCALE)
                
                # Позиция для вставки в сетке
                sub_x = i % 2
                sub_y = i // 2
                
                if n_images == 1:
                    sub_ax = ax
                else:
                    # Создаем вложенный подграфик
                    sub_ax = plt.subplot(n_rows, n_cols * 2, idx * 2 + i + 1)
                
                # Отображаем изображение
                sub_ax.imshow(img, cmap='gray')
                
                # Цвет рамки
                if result['is_correct']:
                    color = 'green' if result['confidence'] > 80 else 'yellow'
                else:
                    color = 'red'
                
                # Добавляем рамку
                for spine in sub_ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(3)
                
                # Подпись
                title = f"True: {result['true_letter']}\nPred: {result['predicted_letter']}\nConf: {result['confidence']:.0f}%"
                sub_ax.set_title(title, fontsize=8, color=color)
                sub_ax.axis('off')
            
            # Заголовок для буквы
            ax.set_title(f"Буква: {letter}", fontsize=12, fontweight='bold')
            ax.axis('off')
        
        plt.suptitle(f'Результаты тестирования датасета\nТочность: {self.total_accuracy:.1f}%', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/results_grid.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_confusion_matrix(self, results):
        """Создание матрицы ошибок"""
        # Получаем все уникальные буквы
        all_letters = sorted(set([r['true_letter'] for r in results] + 
                                 [r['predicted_letter'] for r in results]))
        
        # Создаем матрицу
        n = len(all_letters)
        conf_matrix = np.zeros((n, n))
        
        # Заполняем матрицу
        letter_to_idx = {letter: i for i, letter in enumerate(all_letters)}
        
        for r in results:
            true_idx = letter_to_idx[r['true_letter']]
            pred_idx = letter_to_idx[r['predicted_letter']]
            conf_matrix[true_idx, pred_idx] += 1
        
        # Визуализируем
        fig, ax = plt.subplots(figsize=(20, 16))
        
        im = ax.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
        
        # Настройка отображения
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(all_letters, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(all_letters, fontsize=8)
        
        # Добавляем значения в ячейки
        for i in range(n):
            for j in range(n):
                if conf_matrix[i, j] > 0:
                    text = ax.text(j, i, int(conf_matrix[i, j]),
                                 ha="center", va="center", color="black", fontsize=8)
        
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/confusion_matrix.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def run_test(self, dataset_path):
        """Запуск полного тестирования"""
        # Тестируем датасет
        results = self.test_dataset(dataset_path)
        
        if not results:
            print("❌ Нет результатов для отображения")
            return
        
        # Создаем таблицы
        df, stats, total_accuracy, avg_confidence = self.create_results_table(results)
        self.total_accuracy = total_accuracy
        
        # Выводим статистику
        print(f"\n{'='*70}")
        print("📊 ОБЩАЯ СТАТИСТИКА:")
        print(f"{'='*70}")
        print(f"Всего тестов: {len(results)}")
        print(f"Правильно: {df['is_correct'].sum()}")
        print(f"Ошибок: {(~df['is_correct']).sum()}")
        print(f"Точность: {total_accuracy:.2f}%")
        print(f"Средняя уверенность: {avg_confidence:.2f}%")
        
        print(f"\n📊 СТАТИСТИКА ПО БУКВАМ:")
        print(f"{'='*70}")
        print(stats.to_string())
        
        # Сохраняем результаты
        print(f"\n💾 Результаты сохранены в папку: {self.results_dir}")
        print(f"   - detailed_results.csv - детальные результаты")
        print(f"   - summary_stats.csv - статистика по буквам")
        print(f"   - results_grid.png - сетка с результатами")
        print(f"   - confusion_matrix.png - матрица ошибок")
        
        # Визуализируем
        self.visualize_results_grid(results, max_per_letter=3)
        self.create_confusion_matrix(results)


def main():
    # Путь к датасету
    dataset_path = "dataset"  # Измените на ваш путь
    dataset_path = "/mnt/ntfs/learn_ML/test_classes/Тестовое Python ML,CV/Тестовое_ML/тестовое_ml/dataset/test_unique_only/"  # Измените на ваш путь
    
    # Создаем тестер
    tester = DatasetTester(
        model_path='best_alphabet_model.pth',
        mapping_path='class_mapping.json'
    )
    
    # Запускаем тестирование
    tester.run_test(dataset_path)


if __name__ == "__main__":
    main()