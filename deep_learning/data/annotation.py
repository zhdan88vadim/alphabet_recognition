
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

def _draw_annotations(image, results):
    """Draws frames and captions on the image"""
    # Converting to PIL for Russian text
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    
    # Loading a font with Cyrillic support
    font = _load_font()
    
    for result in results:
        x, y, w, h = result['bbox']
        letter = result['letter']
        confidence = result['confidence']
        
        # Color depending on confidence
        if confidence > 80:
            color = (0, 255, 0)  # Green
        elif confidence > 60:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        
        label = f"{letter} - {confidence:.0f}%"
        
        if font:
            _draw_text_with_background(draw, x, y, label, font, color)
    
    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

def _load_font():
    """Loads a font with Cyrillic support."""
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

def _draw_text_with_background(draw, x, y, text, font, color):
    """Draws text with a background"""
    bbox = draw.textbbox((x, y - 30), text, font=font)
    label_width = bbox[2] - bbox[0]
    label_height = bbox[3] - bbox[1]
    
    # Drawing a background for the text
    draw.rectangle(
        [x, y - label_height - 10, x + label_width + 10, y],
        fill=color
    )
    
    draw.text((x + 5, y - label_height - 5), text, fill=(0, 0, 0), font=font)

def _create_letters_composite(image, results):
    letters_composite = np.zeros_like(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), dtype=np.float64)
    
    for result in results:
        x, y, w, h = result['bbox']
        img_for_save = result.get('img_for_save')
        
        if img_for_save is not None:
            # Resize imaage if needed
            if img_for_save.shape != (h, w):
                img_for_save = cv2.resize(img_for_save, (w, h))
            # Insert image at the required coordintes
            letters_composite[y:y+h, x:x+w] = img_for_save
        
    letters_composite = (letters_composite * 255).astype(np.uint8)
    
    return letters_composite

def visualize_results(image, results, debug_dir):
    vis_image = _draw_annotations(image.copy(), results)

    letters_composite = _create_letters_composite(image.copy(), results)

    composite_with_annotations = _draw_annotations(letters_composite, results)
    
    plt.figure(figsize=(15, 8))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    plt.title('Recognized letters on the original')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(composite_with_annotations, cv2.COLOR_BGR2RGB))
    plt.title('Cut letters in their original positions')
    plt.axis('off')
    
    plt.suptitle(f"Recognized text: {''.join([r['letter'] for r in results])}")
    plt.tight_layout()
    plt.show()

    save_path_1 = f"{debug_dir}/1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    save_path_2 = f"{debug_dir}/2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    save_path_3 = f"{debug_dir}/3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

    plt.imsave(save_path_1, cv2.cvtColor(composite_with_annotations, cv2.COLOR_BGR2RGB), cmap='gray')
    plt.savefig(save_path_2, dpi=150, bbox_inches='tight')
    cv2.imwrite(save_path_3, cv2.cvtColor(composite_with_annotations, cv2.COLOR_BGR2RGB))
