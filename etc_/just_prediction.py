from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
import numpy as np

MODEL_PATH = ""
IMG_PATH = ""

model = load_model(MODEL_PATH)
image_path = IMG_PATH
img = image.load_img(image_path, target_size = (256, 256))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor = preprocess_input(img_tensor)

prediction = model.predict(img_tensor)
category = ["Amerry", "Kano", "Others"]

for i, c in enumerate(category):
    print(c, ":", prediction[0][i])

print(category[np.argmax(prediction)])

