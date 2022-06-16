from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.applications import VGG16

def build_model(model_name="kame1", input_shape=(256,256,3)):
    
    if model_name == "kame1":
        return kame1(input_shape)
    elif model_name == "kame2":
        return kame2(input_shape)
    elif model_name == "kame3":
        return kame3(input_shape)
    elif model_name == "kame4":
        return kame4(input_shape)
    elif model_name == "kame5":
        return kame5(input_shape)
    elif model_name == "kame6":
        return kame6(input_shape)
    elif model_name == "kame7":
        return kame7(input_shape)
    elif model_name == "kame8":
        return kame8(input_shape)
    else:
        print("invalid model.\nMODEL_NAME => kame1 ~ kame8")

def kame1(input_shape=(256,256,3)):
    model = models.Sequential()
    model.add(layers.GaussianNoise(0.2))
    conv_base = VGG16(
        weights = 'imagenet',
        include_top = False,
        input_shape = input_shape
    )
    conv_base.trainable = False
    model.add(conv_base)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(3, activation="softmax"))
    
    model.compile(
        optimizer = optimizers.RMSprop(learning_rate=1e-4),
        loss = "categorical_crossentropy", metrics = ["accuracy"]
    )
    return model

def kame2(input_shape=(128,128,3)):
    model = models.Sequential()
    conv_base = VGG16(
        weights = 'imagenet',
        include_top = False,
        input_shape = input_shape
    )
    conv_base.trainable = False
    model.add(conv_base)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(3, activation="softmax"))
    
    model.compile(
        optimizer = optimizers.RMSprop(learning_rate=1e-4),
        loss = "categorical_crossentropy", metrics = ["accuracy"]
    )
    return model

def kame3(input_shape=(256,256)):
    model = models.Sequential()
    model.add(layers.GaussianNoise(0.2))
    conv_base = VGG16(
        weights = 'imagenet',
        include_top = False,
        input_shape = input_shape
    )
    conv_base.trainable = False
    model.add(conv_base)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(3, activation="softmax"))
    
    model.compile(
        optimizer = optimizers.RMSprop(learning_rate=1e-4),
        loss = "categorical_crossentropy", metrics = ["accuracy"]
    )
    return model


def kame4(input_shape=(256,256,3)):
    model = models.Sequential()
    conv_base = VGG16(
        weights = 'imagenet',
        include_top = False,
        input_shape = input_shape
    )
    conv_base.trainable = False
    model.add(conv_base)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(3, activation="softmax"))
    
    model.compile(
        optimizer = optimizers.RMSprop(learning_rate=1e-4),
        loss = "categorical_crossentropy", metrics = ["accuracy"]
    )
    return model


def kame5(input_shape=(256,256,3)):
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
    model.add(layers.Dense(1024, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(3, activation="softmax"))
    
    model.compile(
        optimizer = optimizers.RMSprop(learning_rate=1e-4),
        loss = "categorical_crossentropy", metrics = ["accuracy"]
    )
    return model


def kame6(input_shape=(256,256,3)):
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


def kame7(input_shape=(224,224,3)):
    model = models.Sequential()
    conv_base = VGG16(
        weights = 'imagenet',
        include_top = False,
        input_shape = input_shape
    )
    conv_base.trainable = False
    model.add(conv_base)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(3, activation="softmax"))
    
    model.compile(
        optimizer = optimizers.RMSprop(learning_rate=1e-4),
        loss = "categorical_crossentropy", metrics = ["accuracy"]
    )
    return model


def kame8(input_shape=(256,256,3)):
    model = models.Sequential()
    conv_base = VGG16(
        weights = 'imagenet',
        include_top = False,
        input_shape = input_shape
    )
    conv_base.trainable = False
    model.add(conv_base)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(3, activation="softmax"))
    
    model.compile(
        optimizer = optimizers.RMSprop(learning_rate=1e-4),
        loss = "categorical_crossentropy", metrics = ["accuracy"]
    )
    return model

