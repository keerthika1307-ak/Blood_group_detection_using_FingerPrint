from tensorflow.keras.preprocessing import image
import numpy as np

def preprocess_image(img, target_size=(64, 64)):
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array
