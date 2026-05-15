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

def save_tensorboard_plots(writer_log_dir, save_dir='images'):
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing import event_accumulator
    import os
    
    os.makedirs(save_dir, exist_ok=True)
    
    ea = event_accumulator.EventAccumulator(writer_log_dir)
    ea.Reload()
    
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        sorted_events = sorted(events, key=lambda x: x.step)
        steps = [e.step for e in sorted_events]
        values = [e.value for e in sorted_events]
        
        plt.figure(figsize=(10, 6))
        
        # ТОЛЬКО ТОЧКИ - честно показывает дискретные измерения
        plt.scatter(steps, values, s=40, alpha=0.8, c='blue', edgecolors='black', linewidth=0.5)
        
        # ИЛИ точки соединены, но маркеры явно видны
        # plt.plot(steps, values, 'o-', linewidth=1, markersize=6, markerfacecolor='blue', 
        #         markeredgecolor='black', alpha=0.8)
        
        plt.xlabel('Step')
        plt.ylabel(tag.split('/')[-1])
        plt.title(tag)
        plt.grid(True, alpha=0.3)
        
        filename = tag.replace('/', '_') + '.png'
        plt.savefig(f'{save_dir}/{filename}', dpi=150, bbox_inches='tight')
        plt.close()
# Вызовите после обучения
save_tensorboard_plots('runs/alphabet_experiment', 'images/training_plots')