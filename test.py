import sys
from inference.predictor import AlphabetPredictor

def main():
    # Пути к файлам модели
    model_path = "best_alphabet_model.pth"
    mapping_path = "class_mapping.json"
    
    # Загружаем предсказатель
    predictor = AlphabetPredictor(model_path, mapping_path)
    
    # Проверяем аргументы командной строки
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "hor_text.png"
        print(f"Используем тестовое изображение: {image_path}")
    
    # Распознаем
    results = predictor.recognize_image(image_path, display=True)
    
    # Выводим результат
    if results:
        recognized_text = "".join([r['letter'] for r in results])
        avg_conf = sum(r['confidence'] for r in results) / len(results)
        
        print(f"\n📝 Распознанный текст: {recognized_text}")
        print(f"📊 Средняя уверенность: {avg_conf:.1f}%")

if __name__ == "__main__":
    main()