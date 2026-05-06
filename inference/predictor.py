
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from PIL import Image, ImageDraw, ImageFont
from models.model import AlphabetRecognizer

class AlphabetPredictor:   
    def __init__(self, model_path, mapping_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model, self.class_names = self._load_model(model_path, mapping_path)

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self.debug_dir = f"debug/debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.debug_dir, exist_ok=True)
    
    def _load_model(self, model_path, mapping_path):
        with open(mapping_path, 'r', encoding='utf-8') as f:
            class_names = json.load(f)
        
        checkpoint = torch.load(model_path, map_location=self.device)

        model = AlphabetRecognizer(num_classes=len(class_names))
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"✅ Модель загружена | Классов: {len(class_names)}")
        return model, class_names
    
    def predict_letter(self, letter_image, return_top5=False):
        _, binary_rgb = cv2.threshold(letter_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        letter_image = Image.fromarray(binary_rgb)
    
        img_tensor = self.transform(letter_image).unsqueeze(0).to(self.device)
        
        img_for_save = img_tensor.squeeze(0).squeeze(0).cpu().numpy()
        img_for_save = (img_for_save - img_for_save.min()) / (img_for_save.max() - img_for_save.min())
        
        plt.imsave(f"{self.debug_dir}/letter_to_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", img_for_save, cmap='gray')

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            
            if return_top5:
                top5_prob, top5_idx = torch.topk(probs, min(5, len(self.class_names)))
                top5 = [(self.class_names[idx.item()], prob.item()) 
                        for idx, prob in zip(top5_idx[0], top5_prob[0])]
                return top5
            
            confidence, predicted = torch.max(probs, 1)
            return self.class_names[predicted.item()], confidence.item() * 100, img_for_save
    
    def preprocess_image(self, image):
        """Предобработка для сегментации"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Бинаризация
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Морфология
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.dilate(binary, kernel, iterations=2)
        
        return gray, binary
    
    def segment_letters(self, image, min_area=300, max_area=7000, aspect_ratio_range=(0.3, 2.5)):
        """Сегментирует буквы на изображении"""
        gray, binary = self.preprocess_image(image)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        letter_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            if area > max_area:
                continue            
            
            x, y, w, h = cv2.boundingRect(cnt)

            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < aspect_ratio_range[0] or aspect_ratio > aspect_ratio_range[1]:
                continue

            padding = 4
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2*padding)
            h = min(image.shape[0] - y, h + 2*padding)
            
            letter_boxes.append({'bbox': (x, y, w, h), 'area': area})
        
        # Сортируем слева направо
        letter_boxes.sort(key=lambda k: k['bbox'][0])
        
        return letter_boxes, gray, binary
    
    def recognize_image(self, image_path, display):
        """Распознает все буквы на изображении"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        letter_boxes, gray, binary = self.segment_letters(image)
        
        results = []
        for i, box in enumerate(letter_boxes, 1):
            x, y, w, h = box['bbox']
            letter_roi = gray[y:y+h, x:x+w]
            
            # Инвертируем для модели
            letter_roi = cv2.bitwise_not(letter_roi)
            
            letter, confidence, img_for_save = self.predict_letter(letter_roi)
            
            results.append({
                'index': i,
                'bbox': (x, y, w, h),
                'letter': letter,
                'confidence': confidence,
                'position': (x + w//2, y + h//2),
                'img_for_save': img_for_save
            })
        
        if display:
            self._visualize_results(image, results, binary)
        
        return results
    
    def _draw_annotations(self, image, results):
        """Рисует рамки и подписи на изображении"""
        # Конвертируем в PIL для русского текста
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)
        
        # Загружаем шрифт с поддержкой кириллицы
        font = self._load_font()
        
        for result in results:
            x, y, w, h = result['bbox']
            letter = result['letter']
            confidence = result['confidence']
            
            # Цвет в зависимости от уверенности
            if confidence > 80:
                color = (0, 255, 0)  # Зеленый
            elif confidence > 60:
                color = (0, 255, 255)  # Желтый
            else:
                color = (0, 0, 255)  # Красный
            
            # Рисуем рамку
            draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
            
            # Подпись на русском
            label = f"{letter} - {confidence:.0f}%"
            
            # Рисуем подпись
            if font:
                self._draw_text_with_background(draw, x, y, label, font, color)
        
        # Конвертируем обратно в OpenCV формат
        return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    def _load_font(self):
        """Загружает шрифт с поддержкой кириллицы"""
        try:
            # Windows
            font_path = "C:/Windows/Fonts/arial.ttf"
            return ImageFont.truetype(font_path, 24)
        except:
            try:
                # Linux
                font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
                return ImageFont.truetype(font_path, 24)
            except:
                return None

    def _draw_text_with_background(self, draw, x, y, text, font, color):
        """Рисует текст с фоном"""
        bbox = draw.textbbox((x, y - 30), text, font=font)
        label_width = bbox[2] - bbox[0]
        label_height = bbox[3] - bbox[1]
        
        # Рисуем фон для подписи
        draw.rectangle(
            [x, y - label_height - 10, x + label_width + 10, y],
            fill=color
        )
        
        draw.text((x + 5, y - label_height - 5), text, fill=(0, 0, 0), font=font)

    def _create_letters_composite(self, image, results):
        """Создает составное изображение из img_for_save"""
        letters_composite = np.zeros_like(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), dtype=np.float64)
        
        for result in results:
            x, y, w, h = result['bbox']
            img_for_save = result.get('img_for_save')
            
            if img_for_save is not None:
                # Изменяем размер img_for_save если нужно
                if img_for_save.shape != (h, w):
                    img_for_save = cv2.resize(img_for_save, (w, h))
                # Вставляем img_for_save на нужные координаты
                letters_composite[y:y+h, x:x+w] = img_for_save
            
        letters_composite = (letters_composite * 255).astype(np.uint8)
        
        return letters_composite

    def _draw_annotations_on_composite(self, composite, results):
        """Рисует рамки и подписи на составном изображении"""
        # composite уже uint8, но возможно 2D, конвертируем в 3-канальный
        if len(composite.shape) == 2:
            composite_color = cv2.cvtColor(composite, cv2.COLOR_GRAY2BGR)
        else:
            composite_color = composite.copy()
        
        # Конвертируем в PIL для русского текста
        composite_pil = Image.fromarray(cv2.cvtColor(composite_color, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(composite_pil)
        
        # Загружаем шрифт
        font = self._load_font()
        
        for result in results:
            x, y, w, h = result['bbox']
            letter = result['letter']
            confidence = result['confidence']
            
            # Цвет в зависимости от уверенности
            if confidence > 80:
                color = (0, 255, 0)  # Зеленый
            elif confidence > 60:
                color = (0, 255, 255)  # Желтый
            else:
                color = (0, 0, 255)  # Красный
            
            # Рисуем рамку
            draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
            
            # Подпись на русском
            label = f"{letter} - {confidence:.0f}%"
            
            self._draw_text_with_background(draw, x, y, label, font, color)
        
        # Конвертируем обратно в OpenCV формат
        return cv2.cvtColor(np.array(composite_pil), cv2.COLOR_RGB2BGR)

    def _visualize_results(self, image, results, binary):
        vis_image = self._draw_annotations(image.copy(), results)

        letters_composite = self._create_letters_composite(image.copy(), results)

        composite_with_annotations = self._draw_annotations(letters_composite, results)
        
        plt.figure(figsize=(15, 8))
        
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.title('Распознанные буквы на оригинале')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(composite_with_annotations, cv2.COLOR_BGR2RGB))
        plt.title('Вырезанные буквы (img_for_save) на исходных позициях')
        plt.axis('off')
        
        plt.suptitle(f"Распознанный текст: {''.join([r['letter'] for r in results])}")
        plt.tight_layout()
        plt.show()

        save_path_1 = f"{self.debug_dir}/1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        save_path_2 = f"{self.debug_dir}/2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        save_path_3 = f"{self.debug_dir}/3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        plt.imsave(save_path_1, cv2.cvtColor(composite_with_annotations, cv2.COLOR_BGR2RGB), cmap='gray')
        plt.savefig(save_path_2, dpi=150, bbox_inches='tight')
        cv2.imwrite(save_path_3, cv2.cvtColor(composite_with_annotations, cv2.COLOR_BGR2RGB))
