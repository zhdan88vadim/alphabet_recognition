import sys
import os
from pathlib import Path
from inference.predictor import AlphabetPredictor

def main():
    model_path = "best_alphabet_model.pth"
    mapping_path = "TODO_REMOVE_THIS"
    
    predictor = AlphabetPredictor(model_path, mapping_path)
    
    # Проверяем аргументы командной строки
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = "hor_text.png"
        print(f"Use test image: {input_path}")
    
    # Проверяем, является ли путь директорией или файлом
    if os.path.isdir(input_path):
        # Распознаем все изображения в папке
        print(f"\n📁 Processing directory: {input_path}")
        
        # Поддерживаемые форматы изображений
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Находим все изображения в папке
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(input_path).glob(f"*{ext}"))
            image_files.extend(Path(input_path).glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"❌ No image files found in {input_path}")
            return
        
        print(f"Found {len(image_files)} image(s):\n")
        
        # Обрабатываем каждое изображение
        all_results = []
        for i, image_path in enumerate(sorted(image_files), 1):
            print(f"{'='*50}")
            print(f"Processing [{i}/{len(image_files)}]: {image_path.name}")
            print(f"{'='*50}")
            
            output_file_name = f"{Path(image_path).stem}__predicted{Path(image_path).suffix}"
            results = predictor.recognize_image(str(image_path), False, output_file_name)
            
            if results:
                recognized_text = "".join([r['letter'] for r in results])
                avg_conf = sum(r['confidence'] for r in results) / len(results)
                
                print(f"📝 Recognized text: {recognized_text}")
                print(f"📊 Average confidence: {avg_conf:.1f}%\n")
                
                all_results.append({
                    'filename': image_path.name,
                    'text': recognized_text,
                    'confidence': avg_conf,
                    'details': results
                })
            else:
                print(f"❌ No text recognized in {image_path.name}\n")
        
        # Выводим сводку по всем изображениям
        print(f"\n{'='*50}")
        print("📊 SUMMARY OF ALL RESULTS")
        print(f"{'='*50}")
        for result in all_results:
            print(f"{result['filename']:<30} -> {result['text']} (conf: {result['confidence']:.1f}%)")
        
    elif os.path.isfile(input_path):
        # Распознаем один файл
        output_file_name = f"{Path(input_path).stem}__predicted{Path(input_path).suffix}"
        results = predictor.recognize_image(input_path, False, output_file_name)
        
        if results:
            recognized_text = "".join([r['letter'] for r in results])
            avg_conf = sum(r['confidence'] for r in results) / len(results)
            
            print(f"\n📝 Recognized text: {recognized_text}")
            print(f"📊 Average confidence: {avg_conf:.1f}%")
        else:
            print("❌ No text recognized")
    else:
        print(f"❌ Path not found: {input_path}")

if __name__ == "__main__":
    main()