from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models, layers, optimizers
import matplotlib.pyplot as plt

INPUT_SHAPE = (128, 128, 3)
PATH_FIGURE = ""
PATH_MODELS = ""


def data_generator(directory, target_size=(128,128), batch_size=20, class_mode='binary'):
    datagen = ImageDataGenerator(rescale=1./255)
    return datagen.flow_from_directory(
        directory,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode
    )

def pre_training(train_data, val_data, test_data):
    model = models.Sequential()
    conv_base = VGG16(
        weights = 'imagenet',
        include_top = False,
        input_shape = INPUT_SHAPE
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

    pre_history = model.fit(
        train_data, epochs = 300, validation_data = val_data
    )
    plot_history(pre_history, history_type="loss")
    plot_history(pre_history, history_type="acc")
    test_loss, test_acc = model.evaluate(test_data)
    results = dict_result(
        pre_history.history["train_loss"], pre_history.history["train_acc"],
        test_loss, test_acc
    )
    save_txt(results)
    model.save("amka2_pretraining.h5")

def fine_tuning(train_data, val_data, test_data):
    model = models.load("amka2_pretraining.h5")
    conv_base = model.layers[0]

    for layer in conv_base.layers:
        if layer.name.startwith("block5"):
            layer.trainable = True
    model.compile(
        optimizer = optimizers.RMSprop(learning_rate=1e-5),
        loss = "binary_crossentropy", metrics = ["accuracy"]
    )
    history = model.fit(
        train_data, epochs = 300, validation_data = val_data
    )
    test_loss, test_acc = model.evaluate(test_data)
    results = dict_result(
        history.history["train_loss"], history.history["train_acc"],
        test_loss, test_acc
    )
    save_txt(results)
    model.save("amka2_fine_tuning.h5")

def dict_result(train_loss, train_acc, test_loss, test_acc):
    results = {
        "train_loss" : train_loss,
        "train_acc" : train_acc,
        "test_loss" : test_loss,
        "test_acc" : test_acc
    }
    return results

def plot_history(history, title="amka2_loss", history_type="loss"):
    val = "val_" + history_type
    
    if len(title.split(".")) == 1:
        title += ".jpg"

    if type(history) == dict:
        h = history
    else:
        h = history.history

    plt.plot(h[history_type])
    plt.plot(h[val])
    plt.title(title)
    plt.ylabel(history_type)
    plt.xlabel("Epochs")
    plt.legend(["Training", "Validation"], loc=0)

def save_txt(result = {}, title="amka2_result"):
    key_list = ["train_loss", "train_acc", "test_loss", "test_acc"]

    if len(title.split(".")) == 1:
        title += ".txt"

    with open(title, "w") as f:
        for key in key_list:
            string = ""
            string += (key + ": ") 
            string += str(result[key])
            string += "\n"
            f.write(string)


