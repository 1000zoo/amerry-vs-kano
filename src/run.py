from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import models, layers, optimizers
import matplotlib.pyplot as plt
from mutil import dict_result, get_input_shape, highlight_string, plot_history, return_shape, save_txt
import amodels
import numpy as np

## 그래프, 모델, loss & acc 텍스트 파일 저장 경로
PATH_FIGURE = "figures/"
PATH_MODELS = "models/"
PATH_TXT = "txtfiles/"

## Data 경로
WINDOW_PATH = "C:/Users/cjswl/Desktop/amerry_vs_kano_data/"
MAC_PATH = "/Users/1000zoo/Desktop/amerry_kano_1/"
DATA_PATH = MAC_PATH

## 선택할 모델 이름 (파일들의 이름 저장에 사용됨)
MODEL_NAME = "kame1"

def data_generator(directory, target_size=(), batch_size=20, class_mode='binary', augmentation=False):
    if augmentation:
        datagen = ImageDataGenerator(
            rescale = 1./255,
            rotation_range=20, shear_range=0.1,
            width_shift_range=0.1, height_shift_range=0.1,
            zoom_range=0.1, horizontal_flip=True, fill_mode='nearest'
        )
    else:
        datagen = ImageDataGenerator(rescale=1./255)

    return datagen.flow_from_directory(
        directory,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode
    )

def pre_training(train_data, val_data, test_data, epochs=100, base="vgg16"):
    input_shape = return_shape(train_data)
    model = amodels.build_model(MODEL_NAME, input_shape)

    pre_history = model.fit(
        train_data, epochs = epochs, validation_data = val_data
    )
    plot_history(pre_history, title=MODEL_NAME+"_pre_train_loss.jpg", history_type="loss")
    plot_history(pre_history, title=MODEL_NAME+"_pre_train_acc.jpg",history_type="accuracy")
    test_loss, test_acc = model.evaluate(test_data)
    results = dict_result(
        pre_history.history["loss"][-1], pre_history.history["accuracy"][-1],
        test_loss, test_acc
    )
    save_txt(results, MODEL_NAME+"_pretraining")
    model.save(PATH_MODELS+MODEL_NAME+"_pretraining.h5")

def fine_tuning(train_data, val_data, test_data, epochs=100):
    model = models.load_model(PATH_MODELS+MODEL_NAME+"_pretraining.h5")
    conv_base = model.layers[0]

    for layer in conv_base.layers:
        if layer.name.startswith("block5"):
            layer.trainable = True
    model.compile(
        optimizer = optimizers.RMSprop(learning_rate=1e-5),
        loss = "binary_crossentropy", metrics = ["accuracy"]
    )
    history = model.fit(
        train_data, epochs = epochs, validation_data = val_data
    )
    plot_history(history, title=MODEL_NAME+"_fine_tuning_loss", history_type="loss")
    plot_history(history, title=MODEL_NAME+"_fine_tuning_acc", history_type="accuracy")
    test_loss, test_acc = model.evaluate(test_data)
    results = dict_result(
        history.history["loss"][-1], history.history["accuracy"][-1],
        test_loss, test_acc
    )
    save_txt(results, MODEL_NAME+"fine_tuning")
    model.save(PATH_MODELS+MODEL_NAME+"_fine_tuning.h5")


def main(epochs=100, target_size=(128,128), batch_size=20):

    train_data = data_generator(DATA_PATH + "project_train", target_size=target_size, augmentation=True)
    val_data = data_generator(DATA_PATH + "project_val", target_size=target_size)
    test_data = data_generator(DATA_PATH + "project_test", target_size=target_size)

    pre_training(train_data, val_data, test_data)
    fine_tuning(train_data, val_data, test_data)


if __name__ == "__main__":
    main()
