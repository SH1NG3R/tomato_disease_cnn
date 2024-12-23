import os
import shutil
import random
from pathlib import Path

def split_dataset(source_dir, train_dir, test_dir, val_dir, train_ratio=0.6, test_ratio=0.3):
    for dir_path in [train_dir, test_dir, val_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    class_folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]
    
    for class_folder in class_folders:
        Path(os.path.join(train_dir, class_folder)).mkdir(exist_ok=True)
        Path(os.path.join(test_dir, class_folder)).mkdir(exist_ok=True)
        Path(os.path.join(val_dir, class_folder)).mkdir(exist_ok=True)
        images = [f for f in os.listdir(os.path.join(source_dir, class_folder)) 
                 if f.lower().endswith('.jpg')]
        random.shuffle(images)
        
        n_train = int(len(images) * train_ratio)
        n_test = int(len(images) * test_ratio)
        
        train_images = images[:n_train]
        test_images = images[n_train:n_train + n_test]
        val_images = images[n_train + n_test:]
        
        for img in train_images:
            shutil.copy2(
                os.path.join(source_dir, class_folder, img),
                os.path.join(train_dir, class_folder, img)
            )
        
        for img in test_images:
            shutil.copy2(
                os.path.join(source_dir, class_folder, img),
                os.path.join(test_dir, class_folder, img)
            )
            
        for img in val_images:
            shutil.copy2(
                os.path.join(source_dir, class_folder, img),
                os.path.join(val_dir, class_folder, img)
            )

base_path = r"D:\Python U\tomate"
source_path = os.path.join(base_path, "dataset")
train_path = os.path.join(base_path, "train")
test_path = os.path.join(base_path, "test")
val_path = os.path.join(base_path, "validar")

split_dataset(source_path, train_path, test_path, val_path)