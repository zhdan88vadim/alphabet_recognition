from pathlib import Path
import sys
from inference.predictor import AlphabetPredictor

def main():
    model_path = "best_alphabet_model.pth"
    mapping_path = "TODO_REMOVE_THIS"
    
    predictor = AlphabetPredictor(model_path, mapping_path)
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "hor_text.png"
        print(f"Use test image: {image_path}")

    output_file_name = f"{Path(image_path).stem}__predicted{Path(image_path).suffix}"
    
    results = predictor.recognize_image(image_path, True, output_file_name)
    
    if results:
        recognized_text = "".join([r['letter'] for r in results])
        avg_conf = sum(r['confidence'] for r in results) / len(results)
        
        print(f"\n📝 Recognized text: {recognized_text}")
        print(f"📊 Average confidence: {avg_conf:.1f}%")

if __name__ == "__main__":
    main()