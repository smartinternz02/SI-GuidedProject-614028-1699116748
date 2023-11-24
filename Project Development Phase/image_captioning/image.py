# image.py
from pathlib import Path
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet201
from PIL import Image

model = DenseNet201()
fe = Model(inputs=model.input, outputs=model.layers[-2].output)

def load_features_from_img_path(image_path: Path, size: tuple[int, int]) -> np.ndarray:
    img_path_str = image_path.absolute()
    img = load_img(img_path_str, target_size=size)
    return load_features_from_img(img, size)

def load_features_from_img(image: Image, size: tuple[int, int]) -> np.ndarray:
    img = image.resize(size)
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    feature_extracted = fe.predict(img)
    return feature_extracted
