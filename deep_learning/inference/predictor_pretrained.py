import sys
from models.pretrained_model import AlphabetRecognizerPretrained
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import json
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import os
import uuid
import numpy as np

from data.annotation import visualize_results
from data.preprocessing import segment_letters
from data.augmentation import ExtractLetterWithMargin, SimpleThinOrThicken, SquarePad

class AlphabetPredictorPretrained:   
    def __init__(self, model_path, mapping_path=None, device='cuda', model_type='pretrained'):
        """
        Args:
            model_path: путь к сохраненной модели (.pth)
            mapping_path: путь к JSON с маппингом классов (опционально)
            device: 'cuda' или 'cpu'
            model_type: 'pretrained' или 'custom'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.model, self.class_names = self._load_model(model_path, mapping_path)
        
        # Размер изображения должен совпадать с тренировочным
        self.image_size = 224  # или 224, смотрите на чем обучена модель
        
        self.transform = self._create_transform()
        
        self.debug_dir = f"debug/debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.debug_dir, exist_ok=True)
        print(f"📁 Debug directory: {self.debug_dir}")
    
    def _create_transform(self):
        """Создает трансформации для инференса (должны совпадать с тренировочными)"""
        return transforms.Compose([
            ExtractLetterWithMargin(margin=4, fill_white=True),
            SquarePad(fill_white=True),
            SimpleThinOrThicken(p=1, strength='strong', is_black_symbol_on_white_background=False),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Диапазон [-1, 1]
        ])
    
    def _load_model(self, model_path, mapping_path):
        """Загружает модель с правильной архитектурой"""
        print(f"📦 Loading model from: {model_path}")
        
        # Загружаем чекпоинт
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Получаем class_names
        if 'class_names' in checkpoint:
            class_names = checkpoint['class_names']
        elif mapping_path and os.path.exists(mapping_path):
            with open(mapping_path, 'r', encoding='utf-8') as f:
                class_names = json.load(f)
        else:
            # Если нет маппинга, пробуем восстановить из модели
            print("⚠️ Warning: No class mapping found, using 33 standard Russian letters")
            class_names = [chr(i) for i in range(ord('А'), ord('Я') + 1)]
            class_names.insert(6, 'Ё')  # Вставляем Ё
            class_names = [c for c in class_names if c != '']  # Убираем пустые
        
        num_classes = len(class_names)
        print(f"📚 Number of classes: {num_classes}")
        
        model = AlphabetRecognizerPretrained(num_classes=num_classes)
        
        # Загружаем веса
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        print(f"✅ Model loaded successfully!")
        print(f"   Classes: {class_names[:5]}... ({len(class_names)} total)")
        
        return model, class_names
    
    def predict_letter(self, letter_image, index, return_top5=False):
        """Предсказывает одну букву"""
        # Инвертируем изображение (если нужно)
        _, binary_rgb = cv2.threshold(letter_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        letter_image_pil = Image.fromarray(binary_rgb)
        
        # Применяем трансформации
        img_tensor = self.transform(letter_image_pil).unsqueeze(0).to(self.device)
        
        # Сохраняем для дебага
        img_for_save = img_tensor.squeeze(0).squeeze(0).cpu().numpy()
        img_for_save = (img_for_save - img_for_save.min()) / (img_for_save.max() - img_for_save.min())
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            
            if return_top5:
                top5_prob, top5_idx = torch.topk(probs, min(5, len(self.class_names)))
                top5 = [(self.class_names[idx.item()], prob.item() * 100) 
                        for idx, prob in zip(top5_idx[0], top5_prob[0])]
                return top5
            
            confidence, predicted = torch.max(probs, 1)
            class_name = self.class_names[predicted.item()]
            confidence_percent = confidence.item() * 100
            
            # Сохраняем обработанное изображение для дебага
            unique_id = uuid.uuid4().hex[:8]
            plt.imsave(f"{self.debug_dir}/letter_{index}_{class_name}_{confidence_percent:.0f}__{unique_id}.png", 
                      img_for_save, cmap='gray')
            
            return class_name, confidence_percent, img_for_save
    
    def recognize_image(self, image_path, display=True, min_confidence=30.0):
        """Распознает все буквы на изображении"""
        print(f"🔍 Processing image: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Сегментируем буквы
        letter_boxes, gray, binary = segment_letters(image)
        print(f"📝 Found {len(letter_boxes)} letter candidates")
        
        results = []
        for i, box in enumerate(letter_boxes, 1):
            x, y, w, h = box['bbox']
            letter_roi = gray[y:y+h, x:x+w]
            
            # Инвертируем для модели
            letter_roi = cv2.bitwise_not(letter_roi)
            
            # Предсказываем букву
            letter, confidence, img_for_save = self.predict_letter(letter_roi, i)
            
            # Фильтруем по уверенности
            if confidence >= min_confidence:
                results.append({
                    'index': i,
                    'bbox': (x, y, w, h),
                    'letter': letter,
                    'confidence': confidence,
                    'position': (x + w//2, y + h//2),
                    'img_for_save': img_for_save
                })
            else:
                print(f"  ⚠️ Letter {i}: low confidence ({confidence:.1f}%), skipped")
        
        # Сортируем по позиции (слева направо)
        results.sort(key=lambda r: r['position'][0])
        
        if display:
            visualize_results(image, results, self.debug_dir)
        
        return results