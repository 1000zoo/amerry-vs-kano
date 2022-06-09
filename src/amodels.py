from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from mutil import plot_history, return_shape, save_txt, dict_result
import sys


def amka1():
    pass

def amka2(input_shape):
    model = models.Sequential()
    conv_base = VGG16(
        weights = 'imagenet',
        include_top = False,
        input_shape = input_shape
    )
    conv_base.trainable = False
    model.add(conv_base)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(1, activation="sigmoid"))
    
    model.compile(
        optimizer = optimizers.RMSprop(learning_rate=1e-4),
        loss = "binary_crossentropy", metrics = ["accuracy"]
    )
    
    
    return model