
from data.annotation import visualize_results
from data.preprocessing import segment_letters
import torch
from torchvision import transforms
from PIL import Image
import json
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import os
from PIL import Image
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
        
        print(f"✅ Model loaded | Classes: {len(class_names)}")
        return model, class_names
    
    def predict_letter(self, letter_image, index, return_top5=False):
        _, binary_rgb = cv2.threshold(letter_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        letter_image = Image.fromarray(binary_rgb)
    
        img_tensor = self.transform(letter_image).unsqueeze(0).to(self.device)
        
        img_for_save = img_tensor.squeeze(0).squeeze(0).cpu().numpy()
        img_for_save = (img_for_save - img_for_save.min()) / (img_for_save.max() - img_for_save.min())
        
        plt.imsave(f"{self.debug_dir}/letter_to_model_{index}.png", img_for_save, cmap='gray')

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
    
    def recognize_image(self, image_path, display):
        """Recognizes all letters in an image"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        letter_boxes, gray, binary = segment_letters(image)
        
        results = []
        for i, box in enumerate(letter_boxes, 1):
            x, y, w, h = box['bbox']
            letter_roi = gray[y:y+h, x:x+w]
            
            # Invert for the model
            letter_roi = cv2.bitwise_not(letter_roi)
            
            letter, confidence, img_for_save = self.predict_letter(letter_roi, i)
            
            results.append({
                'index': i,
                'bbox': (x, y, w, h),
                'letter': letter,
                'confidence': confidence,
                'position': (x + w//2, y + h//2),
                'img_for_save': img_for_save
            })
        
        if display:
            visualize_results(image, results, self.debug_dir)
        
        return results
    