from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from mutil import plot_history, return_shape, save_txt, dict_result
import sys

def build_model(model_name="kame1", input_shape=(256,256)):
    
    if model_name == "kame1":
        return kame1(input_shape)
    elif model_name == "kame2":
        pass
    elif model_name == "kame3":
        pass
    elif model_name == "kame4":
        pass
    elif model_name == "kame5":
        pass
    elif model_name == "kame6":
        pass
    elif model_name == "kame7":
        pass
    elif model_name == "kame8":
        pass

def kame1(input_shape=(256,256)):
    model = models.Sequential()
    conv_base = VGG16(
        weights = 'imagenet',
        include_top = False,
        input_shape = input_shape
    )
    conv_base.trainable = False
    model.add(conv_base)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.1))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(3, activation="softmax"))
    
    model.compile(
        optimizer = optimizers.RMSprop(learning_rate=1e-4),
        loss = "categorical_crossentropy", metrics = ["accuracy"]
    )
    return model