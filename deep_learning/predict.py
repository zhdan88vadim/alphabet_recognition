import cv2
from models.model import AlphabetRecognizer
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, models
from PIL import Image
import json
import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont


image_path = "hor_text.png"
# image_path = "private_local/test_letters_my0.png"
# image_path = "private_local/test_my.png"


class MultiLetterRecognizerDebug:
    def __init__(self, model_path='best_alphabet_model.pth', mapping_path='class_mapping.json'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        self.debug_dir = f"debug/debug_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.debug_dir, exist_ok=True)        
        print(f"Debug output folder: {self.debug_dir}\n")
        
        self.model, self.class_names = self._load_model(model_path, mapping_path)
        
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, model_path, mapping_path):
        with open(mapping_path, 'r', encoding='utf-8') as f:
            class_names = json.load(f)
        
        print(f"Count of classes: {len(class_names)}")
        checkpoint = torch.load(model_path, map_location=self.device)

        num_classes = len(class_names)
        model = AlphabetRecognizer()

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"Model loaded: {model_path}")
        if 'val_acc' in checkpoint:
            print(f"   Better val_acc: {checkpoint['val_acc']:.2f}%")
        print(f"   Classes: {', '.join(class_names[:10])}...")
        
        return model, class_names
    
    def preprocess_image(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)            

        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )
        
        cv2.imwrite(f"{self.debug_dir}/0_binary_mask_before_morph.png", binary)

        kernel = np.ones((3, 3), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        cv2.imwrite(f"{self.debug_dir}/0_binary_mask.png", binary)

        kernel = np.ones((3,3),np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=2) 

        cv2.imwrite(f"{self.debug_dir}/0_binary_mask_after_dilate.png", binary)
        
        return gray, binary
    
    def segment_letters(self, image, min_area=300, sort_left_to_right=True, aspect_ratio_range=(0.7, 1.3)):
        gray, binary = self.preprocess_image(image)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        letter_boxes = []
        
        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            
            if area < min_area:
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
            
            letter_boxes.append({
                'bbox': (x, y, w, h),
                'contour': cnt,
                'area': area,
                'original_idx': idx
            })
        
        if sort_left_to_right:
            letter_boxes.sort(key=lambda k: k['bbox'][0])
        
        print(f"🔍 Found counters: {len(letter_boxes)}")
        
        vis_contours = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for i, box in enumerate(letter_boxes):
            x, y, w, h = box['bbox']
            cv2.rectangle(vis_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(vis_contours, f"#{i+1}", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # cv2.imwrite(f"{self.debug_dir}/1_detected_contours.png", vis_contours)
        
        return letter_boxes, gray, binary
        

    def _nms_merge_boxes(self, boxes, iou_threshold=0.5):
        """Объединяет боксы с помощью Non-Maximum Suppression
        
        Args:
            boxes: список словарей с ключом 'bbox'
            iou_threshold: порог IoU для объединения
        
        Returns:
            список объединенных боксов
        """
        if not boxes:
            return []
        
        # Извлекаем все боксы
        bbox_list = [box['bbox'] for box in boxes]
        
        # Конвертируем в формат [x1, y1, x2, y2] для удобства
        rects = []
        for box in boxes:
            x, y, w, h = box['bbox']
            rects.append([x, y, x + w, y + h])
        
        # Применяем NMS
        pick = []
        rects = np.array(rects)
        x1 = rects[:, 0]
        y1 = rects[:, 1]
        x2 = rects[:, 2]
        y2 = rects[:, 3]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            overlap = (w * h) / areas[idxs[:last]]
            
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > iou_threshold)[0])))
        
        # Возвращаем выбранные боксы
        return [boxes[i] for i in pick]

    def recognize_letter(self, letter_image, letter_idx):
        # Сохраняем исходный ROI
        # cv2.imwrite(f"{self.debug_dir}/2_letter_{letter_idx}_roi.png", letter_image)
        
        print(f"   📐 Размер ROI: {letter_image.shape[1]}x{letter_image.shape[0]}")

        _, binary_rgb = cv2.threshold(letter_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # binary_rgb = cv2.cvtColor(binary_rgb, cv2.COLOR_GRAY2RGB)
                
        img_pil = Image.fromarray(binary_rgb)
        
        # Применяем трансформации
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        
        # Показываем что идет в модель
        print(f"   Вход в модель:")
        print(f"      - Размер тензора: {img_tensor.shape}")
        print(f"      - Min значение: {img_tensor.min().item():.3f}")
        print(f"      - Max значение: {img_tensor.max().item():.3f}")
        print(f"      - Mean значение: {img_tensor.mean().item():.3f}")
        
        # Сохраняем преобразованное изображение для проверки
        # img_for_save = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_for_save = img_tensor.squeeze(0).squeeze(0).cpu().numpy()  # [64, 64]
        img_for_save = (img_for_save - img_for_save.min()) / (img_for_save.max() - img_for_save.min())
        plt.imsave(f"{self.debug_dir}/3_letter_{letter_idx}_to_model.png", img_for_save, cmap='gray')


        # plt.figure(figsize=(15, 8))
        # plt.imshow(cv2.cvtColor(img_for_save, cv2.COLOR_GRAY2RGB), cmap='gray')
        # plt.title('🎯 Финальный результат', fontsize=14, fontweight='bold')
        # plt.axis('off')
        # plt.show()
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            
            top5_prob, top5_idx = torch.topk(probs, min(5, len(self.class_names)))
            top5_list = []
            print(f"   🎯 Топ-5 предсказаний:")
            for idx, (prob, class_idx) in enumerate(zip(top5_prob[0], top5_idx[0])):
                letter = self.class_names[class_idx.item()]
                conf = prob.item() * 100
                top5_list.append((letter, conf))
                marker = "✅" if idx == 0 else ""
                print(f"      {marker} {idx+1}. {letter}: {conf:.2f}%")
        
        predicted_letter = self.class_names[predicted.item()]
        confidence_score = confidence.item() * 100

        print(predicted_letter)
        
        return predicted_letter, confidence_score, top5_list
    
    def recognize_image(self, image_path, display=True, save_path=None):
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Can not load image: {image_path}")
            return None

        original = image.copy()
        print(f"\n{'='*70}")
        print(f"FIle: {os.path.basename(image_path)}")
        print(f"Image size: {image.shape[1]}x{image.shape[0]}")
        print(f"{'='*70}\n")
        
        # Сохраняем оригинал
        # cv2.imwrite(f"{self.debug_dir}/0_original_image.png", image)
        
        letter_boxes, gray, binary = self.segment_letters(image)
        
        print(f"\n🔍 Найдено букв: {len(letter_boxes)}")
        print("-" * 70)
        
        results = []
        
        for i, letter_info in enumerate(letter_boxes, 1):
            x, y, w, h = letter_info['bbox']
            
            print(f"\n🔤 БУКВА #{i}:")
            print(f"   📍 Позиция: x={x}, y={y}, w={w}, h={h}")
            
            letter_roi = gray[y:y+h, x:x+w]
            # letter_roi = binary[y:y+h, x:x+w]

            letter_roi = cv2.bitwise_not(letter_roi)

            predicted_letter, confidence, top5 = self.recognize_letter(letter_roi, i)
            
            results.append({
                'index': i,
                'bbox': (x, y, w, h),
                'predicted_letter': predicted_letter,
                'confidence': confidence,
                'top5': top5
            })
            
            status = "✅" if confidence > 80 else "⚠️" if confidence > 60 else "❌"
            print(f"   {status} РЕЗУЛЬТАТ: {predicted_letter} (уверенность: {confidence:.1f}%)")
        
        print("\n" + "="*70)
        print("📋 СВОДКА ПО ВСЕМ БУКВАМ:")
        print("="*70)
        for r in results:
            status = "✅" if r['confidence'] > 80 else "⚠️" if r['confidence'] > 60 else "❌"
            print(f"  {status} Буква {r['index']}: {r['predicted_letter']} ({r['confidence']:.1f}%)")
        
        if display or save_path:
            self._visualize_results(original, results, binary, display, save_path)
        
        return results
    
    def _visualize_results(self, original_image, results, binary, display=True, save_path=None):
        vis_image = original_image.copy()
        
        # Загружаем шрифт с поддержкой кириллицы
        # Путь к шрифту (выберите один из вариантов)
        try:
            # Windows
            font_path = "C:/Windows/Fonts/arial.ttf"
            font = ImageFont.truetype(font_path, 24)
            small_font = ImageFont.truetype(font_path, 18)
        except:
            try:
                # Linux
                font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
                font = ImageFont.truetype(font_path, 24)
                small_font = ImageFont.truetype(font_path, 18)
            except:
                # Если шрифт не найден, используем стандартный OpenCV (только латиница)
                use_pil = False
                font = None
        
        # Конвертируем в PIL для русского текста
        vis_image_pil = Image.fromarray(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(vis_image_pil)
        
        for result in results:
            x, y, w, h = result['bbox']
            letter = result['predicted_letter']
            confidence = result['confidence']
            
            # Цвет в зависимости от уверенности
            if confidence > 80:
                color = (0, 255, 0)  # Зеленый
            elif confidence > 60:
                color = (0, 255, 255)  # Желтый
            else:
                color = (0, 0, 255)  # Красный
            
            draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
            
            # Подпись на русском
            label = f"{letter} - {confidence:.0f}%"
            
            # Получаем размер текста
            if font:
                bbox = draw.textbbox((x, y - 30), label, font=font)
                label_width = bbox[2] - bbox[0]
                label_height = bbox[3] - bbox[1]
                
                # Рисуем фон для подписи
                draw.rectangle(
                    [x, y - label_height - 10, x + label_width + 10, y],
                    fill=color
                )
                
                # Рисуем текст белым цветом
                draw.text((x + 5, y - label_height - 5), label, fill=(255, 255, 255), font=font)
        
        # Конвертируем обратно в OpenCV формат
        vis_image = cv2.cvtColor(np.array(vis_image_pil), cv2.COLOR_RGB2BGR)

        if display:
            fig, axes = plt.subplots(3, 1, figsize=(30, 60))
            
            axes[0].imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
            axes[0].set_title('📝 Распознанные буквы', fontsize=14, fontweight='bold')
            axes[0].axis('off')
            
            axes[1].imshow(binary, cmap='gray')
            axes[1].set_title('🔍 Бинарная маска', fontsize=14)
            axes[1].axis('off')
            
            # Статистика
            axes[2].axis('off')
            stats_text = "📊 Статистика:\n\n"
            for r in results:
                conf_color = "✅" if r['confidence'] > 80 else "⚠️" if r['confidence'] > 60 else "❌"
                stats_text += f"{conf_color} Буква {r['index']}: {r['predicted_letter']} ({r['confidence']:.1f}%)\n"
            
            if results:
                recognized_text = "".join([r['predicted_letter'] for r in results])
                avg_conf = np.mean([r['confidence'] for r in results])
                stats_text += f"\n💬 Распознанный текст: {recognized_text}"
                stats_text += f"\n📈 Средняя уверенность: {avg_conf:.1f}%"
            
            axes[2].text(0.1, 0.9, stats_text, transform=axes[2].transAxes,
                        fontsize=12, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.suptitle('РЕЗУЛЬТАТЫ РАСПОЗНАВАНИЯ', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"💾 Визуализация сохранена: {save_path}")
            
            # plt.show()
        else:
            if save_path:
                cv2.imwrite(save_path, vis_image)
                print(f"💾 Изображение сохранено: {save_path}")


def main():
    recognizer = MultiLetterRecognizerDebug(
        model_path='best_alphabet_model.pth',
        mapping_path='class_mapping.json'
    )
    
    results = recognizer.recognize_image(
        image_path,
        display=True,
        save_path='recognition_result.png'
    )
    
    if results:
        print("\n" + "="*70)
        print("📋 ИТОГИ:")
        print("="*70)
        recognized_text = "".join([r['predicted_letter'] for r in results])
        avg_confidence = np.mean([r['confidence'] for r in results])
        print(f"Распознанный текст: {recognized_text}")
        print(f"Средняя уверенность: {avg_confidence:.1f}%")
        print("="*70)
        print(f"\n💾 Все отладочные файлы сохранены в папку: {recognizer.debug_dir}/")


if __name__ == "__main__":
    main()