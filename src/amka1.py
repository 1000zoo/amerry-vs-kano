"""
Amerry vs. Kano
05/25
Dogs-and-cats copy
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers, optimizers
import matplotlib.pyplot as plt

dirpath = "/content/drive/MyDrive/ann_project/amery-kano-dachshund-other"

train_dir = dirpath + "/project_train"
validation_dir = dirpath + "/project_val"
test_dir = dirpath + "/project_test"

# data augmentation 추가
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

# model definition
input_shape = [150, 150, 3] # as a shape of image
def build_model():
    model=models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    # compile
    model.compile(optimizer=optimizers.RMSprop(lr=1e-3),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

# main loop without cross-validation
import time
starttime=time.time()
num_epochs = 30
model = build_model()
history = model.fit_generator(train_generator,
                    epochs=num_epochs,
                    validation_data=validation_generator)

# evaluation
train_loss, train_acc = model.evaluate_generator(train_generator)
test_loss, test_acc = model.evaluate_generator(test_generator)
print('train_acc:', train_acc)
print('test_acc:', test_acc)
print("elapsed time (in sec): ", time.time()-starttime)

# visualization
def plot_acc(h, title="accuracy"):
    plt.plot(h.history['accuracy'])
    plt.plot(h.history ['val_accuracy'])
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)

def plot_loss(h, title="loss"):
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)

# import os
# file_name = os.path.basename(__file__)
# ch = ".py"

# for c in ch:
#     file_name = file_name.replace(c, "")
# wlist = file_name.split("_")
# qnum = wlist[-1]
plot_loss(history)

# saving the model
model_path = "/content/drive/MyDrive/ann_project/result_model"
model.save(model_path + 'amerry_vs_kano.h5')
fig_path = "/content/drive/MyDrive/ann_project/result_figure/"
plt.savefig(fig_path + 'amerry_vs_kano_loss.png')
plt.clf()
plot_acc(history)
plt.savefig(fig_path + 'amerry_vs_kano_accuracy.png')
