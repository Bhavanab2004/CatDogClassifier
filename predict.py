import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = 'cat_dog_model.h5'
IMG_SIZE = 128

if len(sys.argv) != 2:
    print('Usage: python predict.py <image_path>')
    sys.exit(1)

img_path = sys.argv[1]

# Load and preprocess image
img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
x = image.img_to_array(img)
x = x / 255.0
x = np.expand_dims(x, axis=0)

# Load model and predict
model = load_model(MODEL_PATH)
pred = model.predict(x)[0][0]

if pred > 0.5:
    print('Dog')
else:
    print('Cat')
