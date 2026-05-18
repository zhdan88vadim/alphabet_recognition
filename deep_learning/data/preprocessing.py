
import cv2
import numpy as np

def preprocess_image(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.dilate(binary, kernel, iterations=2)
    
    return gray, binary


def segment_letters(image, min_area=450, max_area=7000, aspect_ratio_range=(0.3, 2.5)):
    """Сегментирует буквы на изображении"""
    gray, binary = preprocess_image(image)
    
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
    
    # Sort from left to right
    letter_boxes.sort(key=lambda k: k['bbox'][0])
    
    return letter_boxes, gray, binary
