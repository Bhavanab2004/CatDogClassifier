import os
from PIL import Image

# Paths
DATASET_DIR = 'PetImages'
CATEGORIES = ['Dog', 'Cat']
IMG_SIZE = 128  # Resize images to 128x128

# Remove corrupted images and resize
for category in CATEGORIES:
    folder = os.path.join(DATASET_DIR, category)
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            with Image.open(file_path) as img:
                img = img.convert('RGB')
                img = img.resize((IMG_SIZE, IMG_SIZE))
                img.save(file_path)
        except Exception as e:
            print(f'Removing corrupted file: {file_path}')
            os.remove(file_path)
print('Dataset cleaned and resized.')
