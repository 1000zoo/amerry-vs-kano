from tensorflow.keras import models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
import numpy as np

MODEL_PATH = "models/amka2_categorical_fine_tuning.h5"
# IMG_PATH = "/Users/1000zoo/Desktop/ann-project/amnewenewnewnew/IMG_5610.jpeg"
IMG_PATH = "/Users/1000zoo/Desktop/ann-project/kanewnewnewnew/IMG_5585.jpeg"
# IMG_PATH = "/Users/1000zoo/Documents/prog/data_files/dogs_and_cats/train/dogs/dog.5112.jpg"

# def data_generator(directory, target_size=(64,64), batch_size=20, class_mode='categorical'):
#     datagen = ImageDataGenerator(rescale=1./255)
#     return datagen.flow_from_directory(
#         directory,
#         target_size=target_size,
#         batch_size=batch_size,
#         class_mode=class_mode
#     )

model = models.load_model(MODEL_PATH)

# model.compile(loss="categorical_crossentropy", optimizer=optimizers.RMSprop(learning_rate=1e-5), metrics=["accuracy"])

# images = data_generator(IMG_PATH)

# prediction = model.predict(images)
# print(prediction)

image_path = IMG_PATH
img = image.load_img(image_path, target_size = (64, 64))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor = preprocess_input(img_tensor)

prediction = model.predict(img_tensor)
model.summary()
category = ["Amerry", "Kano", "Others"]
plt.imshow(img)
print(prediction)
print(category[np.argmax(prediction)])
plt.show()
