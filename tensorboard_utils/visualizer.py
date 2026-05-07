import torchvision

def log_transformed_images(writer, dataset, num_samples=8, tag="train_transforms"):
    """Визуализирует изображения после трансформаций"""
    
    # Берем несколько примеров
    images, labels = [], []
    for i in range(min(num_samples, len(dataset))):
        img, label = dataset[i]
        images.append(img)
        labels.append(label)
    
    # Создаем сетку изображений
    img_grid = torchvision.utils.make_grid(images, nrow=6, normalize=True)
    
    # Логируем в TensorBoard
    writer.add_image(f'{tag}/images', img_grid, 0)
    
    # Также логируем отдельно каждое изображение
    for i, (img, label) in enumerate(zip(images, labels)):
        writer.add_image(f'{tag}/sample_{i}', img, 0)
        writer.add_text(f'{tag}/labels', f"Class: {label}", i)